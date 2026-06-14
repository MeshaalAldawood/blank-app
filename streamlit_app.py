import re
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

BASE_URL = "https://haraj.com.sa"
DEFAULT_QUERIES = "برونكو 2021\nبرونكو 2022\nبرونكو 2023"

BAD_WORDS = [
    "تشليح", "قطع", "قطع غيار", "مكينة", "قير", "ايرباق", "ارباق", "جنط", "جنوط",
    "كفر", "كفرات", "صدام", "رفرف", "سطبات", "شمعة", "فلتر", "فحمات", "دعاسة",
    "تلبيسة", "اكسسوار", "اكسسوارات", "مصدوم", "حادث", "للبيع قطع", "مساعد", "هوب"
]

# English + Arabic terms that commonly appear on Haraj Bronco listings.
SYNONYMS = {
    "برونكو": ["برونكو", "برنكو", "bronco"],
    "فورد": ["فورد", "ford"],
}

@dataclass
class Listing:
    query: str
    title: str
    price: Optional[int]
    city: str
    age: str
    link: str
    relevance: float


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه").replace("ى", "ي")
    text = re.sub(r"[\u064B-\u065F]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def query_terms(query: str) -> List[str]:
    return [normalize(w) for w in query.split() if w.strip()]


def has_term(title: str, term: str) -> bool:
    title_n = normalize(title)
    options = SYNONYMS.get(term, [term])
    return any(normalize(opt) in title_n for opt in options)


def relevance_score(title: str, query: str) -> float:
    terms = query_terms(query)
    if not terms:
        return 0.0
    matched = sum(1 for term in terms if has_term(title, term))
    return matched / len(terms)


def looks_bad(title: str) -> bool:
    title_n = normalize(title)
    return any(normalize(w) in title_n for w in BAD_WORDS)


def extract_price(text: str) -> Optional[int]:
    # Haraj prices usually appear as plain 5-6 digit SAR amounts. Avoid huge phone/listing ids.
    nums = re.findall(r"(?<!\d)(\d{2,3}(?:,\d{3})+|\d{5,6})(?!\d)", text.replace("٬", ","))
    candidates = []
    for n in nums:
        try:
            value = int(n.replace(",", ""))
        except ValueError:
            continue
        if 10_000 <= value <= 1_000_000:
            candidates.append(value)
    return candidates[-1] if candidates else None


def build_search_url(query: str, page: int) -> str:
    encoded = quote(query)
    # Tag pages are more focused than generic /search pages for car queries.
    if page <= 1:
        return f"{BASE_URL}/tags/{encoded}/"
    return f"{BASE_URL}/tags/{encoded}/?page={page}"


def request_page(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; PersonalPriceResearch/1.0)",
        "Accept-Language": "ar,en;q=0.9",
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.text


def parse_search_results(html: str, query: str) -> List[Listing]:
    soup = BeautifulSoup(html, "html.parser")
    results: List[Listing] = []
    seen_links = set()

    # Haraj listing links often contain /111...; this keeps the parser robust if CSS classes change.
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/111" not in href:
            continue
        link = href if href.startswith("http") else BASE_URL + href
        if link in seen_links:
            continue
        seen_links.add(link)

        text = a.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 8:
            continue

        # Title is usually the most meaningful line before city/time/price.
        parts = [p.strip() for p in re.split(r"\s{2,}|\n", text) if p.strip()]
        title = parts[0] if parts else text[:120]
        price = extract_price(text)
        score = relevance_score(title, query)

        city = ""
        age = ""
        common_cities = ["الرياض", "جدة", "جده", "الدمام", "الخبر", "مكة", "مكه", "المدينة", "بريدة", "خميس مشيط", "صبيا", "الطائف"]
        for city_name in common_cities:
            if city_name in text:
                city = city_name
                break
        age_match = re.search(r"(قبل\s+[^\d]{0,12}\d*\s*\w+|اليوم|أمس|امس|yesterday|\d+\s*(hr|hour|day|days|wk|mo))", text, re.I)
        if age_match:
            age = age_match.group(0)

        results.append(Listing(query, title, price, city, age, link, score))

    return results


def scrape_query(query: str, min_relevance: float, max_irrelevant_streak: int, max_pages: int, delay: float) -> pd.DataFrame:
    rows: List[Dict] = []
    irrelevant_streak = 0
    seen = set()

    for page in range(1, max_pages + 1):
        url = build_search_url(query, page)
        html = request_page(url)
        listings = parse_search_results(html, query)
        if not listings:
            break

        for item in listings:
            if item.link in seen:
                continue
            seen.add(item.link)

            relevant = item.relevance >= min_relevance and not looks_bad(item.title)
            if relevant:
                rows.append({
                    "query": item.query,
                    "title": item.title,
                    "price_sar": item.price,
                    "city": item.city,
                    "age": item.age,
                    "relevance": round(item.relevance, 2),
                    "link": item.link,
                })
                irrelevant_streak = 0
            else:
                irrelevant_streak += 1

            if irrelevant_streak >= max_irrelevant_streak:
                return pd.DataFrame(rows)

        time.sleep(delay)

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.dropna(subset=["price_sar"]).copy()
    if clean.empty:
        return pd.DataFrame(columns=["query", "priced_listings", "median", "min", "max", "fair_listing", "fast_sale", "negotiation_floor"])
    clean["price_sar"] = clean["price_sar"].astype(int)
    grouped = clean.groupby("query")["price_sar"]
    summary = grouped.agg(priced_listings="count", median="median", min="min", max="max").reset_index()
    summary["fair_listing"] = (summary["median"] * 1.01).round(-3).astype(int)
    summary["fast_sale"] = (summary["median"] * 0.92).round(-3).astype(int)
    summary["negotiation_floor"] = (summary["median"] * 0.88).round(-3).astype(int)
    for col in ["median", "min", "max"]:
        summary[col] = summary[col].round(0).astype(int)
    return summary


st.set_page_config(page_title="Haraj Car Price Finder", page_icon="🚙", layout="wide")
st.title("Haraj Car Price Finder")
st.caption("Personal market research tool. Searches focused Haraj tag pages, stops when results become irrelevant, and exports prices.")

queries_text = st.text_area("Search queries — one per line", DEFAULT_QUERIES, height=120)

with st.expander("Settings", expanded=False):
    min_relevance = st.slider("Minimum title relevance", 0.3, 1.0, 0.65, 0.05)
    max_irrelevant_streak = st.slider("Stop after this many irrelevant listings in a row", 5, 50, 18, 1)
    max_pages = st.slider("Maximum pages per query", 1, 15, 6, 1)
    delay = st.slider("Delay between pages, seconds", 0.5, 5.0, 1.5, 0.5)

if st.button("Search prices", type="primary"):
    queries = [q.strip() for q in queries_text.splitlines() if q.strip()]
    all_frames = []
    progress = st.progress(0)
    status = st.empty()

    for i, q in enumerate(queries, start=1):
        status.write(f"Searching: {q}")
        try:
            df_q = scrape_query(q, min_relevance, max_irrelevant_streak, max_pages, delay)
            all_frames.append(df_q)
        except Exception as e:
            st.error(f"Could not search {q}: {e}")
        progress.progress(i / len(queries))

    if all_frames:
        df = pd.concat(all_frames, ignore_index=True).drop_duplicates("link")
    else:
        df = pd.DataFrame()

    if df.empty:
        st.warning("No relevant results found. Try lowering relevance or using a broader query like: برونكو 2021")
    else:
        summary = summarize(df)
        st.subheader("Price summary")
        st.dataframe(summary, use_container_width=True)

        st.subheader("Listings")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download listings CSV", csv, "haraj_car_prices.csv", "text/csv")

        summary_csv = summary.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download summary CSV", summary_csv, "haraj_price_summary.csv", "text/csv")

st.info("Note: Haraj asking prices are not final sale prices. Fair sale estimate usually needs mileage, trim, accident history, and negotiation discount.")
