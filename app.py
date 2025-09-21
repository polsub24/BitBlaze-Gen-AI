# app.py
import streamlit as st
import google.generativeai as genai
import requests
import os
import json
import re
from dotenv import load_dotenv

# ----------------------
# 🔑 API Key Setup
# ----------------------
load_dotenv()

# Primary env names supported (backwards-friendly)
gemini_key = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GEM_KEY")
    or os.getenv("gem_key")
)
fact_check_key = (
    os.getenv("FACTCHECK_API_KEY")
    or os.getenv("FACT_CHECK_KEY")
    or os.getenv("FACTCHECK")
)
newsapi_key = (
    os.getenv("NEWSAPI_KEY")
    or os.getenv("NEWS_API_KEY")
    or os.getenv("NEWSAPI")
)

# Configure Gemini client if key present
if gemini_key:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
else:
    model = None  # handle gracefully later

# ----------------------
# 🔗 Trusted Sources (kept comprehensive; UI remains unchanged)
# ----------------------
DOMAIN_SOURCES = {
    "Health": ["WHO", "CDC", "NIH", "ICMR", "PubMed", "MedlinePlus"],
    "Politics": ["Election Commission", "UN", "Reuters", "BBC", "Press Information Bureau India"],
    "Science": ["Nature", "Science", "arXiv", "IEEE", "PLOS"],
    "Finance": ["RBI", "SEBI", "World Bank", "IMF", "Economic Times"],
    "Climate": ["IPCC", "NASA Climate", "NOAA", "UNEP", "MoEFCC India"],
    "Technology": ["IEEE", "ACM", "Nature Tech", "ScienceDirect"],
    "General": ["Reuters", "BBC", "AP News", "The Hindu"]
}

# ----------------------
# 🔍 Fact Check API (Google Fact Check Tools)
# ----------------------
def fact_check_search(query: str):
    """Query Google Fact Check Tools. Returns parsed JSON or {} on failure."""
    if not fact_check_key:
        return {}
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": fact_check_key}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
        else:
            return {}
    except Exception:
        return {}

# ----------------------
# 🌐 News / Web Verification (NewsAPI - optional)
# ----------------------
def web_verify(claim: str):
    """Search news (NewsAPI). Returns list of top URLs or [] if unavailable."""
    if not newsapi_key:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {"q": claim, "apiKey": newsapi_key, "pageSize": 5, "sortBy": "relevancy"}
    try:
        r = requests.get(url, params=params, timeout=10)
        j = r.json()
        articles = j.get("articles", [])
        return [a.get("url") for a in articles[:3] if a.get("url")]
    except Exception:
        return []

# ----------------------
# Helper: robust model generation + text extraction
# ----------------------
def _generate_with_model(prompt: str):
    """Call the Gemini model if configured. Return (text, error_message)"""
    if model is None:
        return None, "Model not configured (missing GEMINI_API_KEY)."
    try:
        resp = model.generate_content(prompt)
    except Exception as e:
        return None, f"Model request failed: {e}"

    # Extract textual content robustly
    text = None
    if hasattr(resp, "text"):
        text = resp.text
    elif hasattr(resp, "content"):
        try:
            if isinstance(resp.content, list):
                pieces = []
                for it in resp.content:
                    if isinstance(it, dict):
                        pieces.append(it.get("text", str(it)))
                    else:
                        pieces.append(str(it))
                text = "\n".join(pieces)
            elif isinstance(resp.content, dict):
                text = resp.content.get("text", str(resp.content))
            else:
                text = str(resp.content)
        except Exception:
            text = str(resp.content)
    else:
        text = str(resp)

    return text, None

def _extract_json_from_text(text: str):
    """Try to locate a JSON object inside text and parse it. Returns dict or None."""
    if not text:
        return None
    # Try direct JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    # Try finding first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        snippet = m.group(0)
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    # No parseable JSON found
    return None

# ----------------------
# ✅ Unified Claim Verification (robust)
# ----------------------
def verify_claim(claim: str, domain: str = "Health"):
    """
    Verify a claim using:
    1) Google Fact Check Tools
    2) News search + Gemini model
    3) Fallback to trusted domain-specific sources + Gemini
    Returns a dict with keys: claim, domain, status, confidence (0-1), explanation, sources (list)
    """
    # normalize domain key (preserve capitalized keys used by UI)
    domain_key = domain.title() if isinstance(domain, str) else "General"
    if domain_key not in DOMAIN_SOURCES:
        domain_key = "General"

    # 1) FactCheck tools
    try:
        fc = fact_check_search(claim)
        if fc and isinstance(fc, dict) and "claims" in fc and len(fc["claims"]) > 0:
            claim_data = fc["claims"][0]
            review = claim_data.get("claimReview", [{}])[0]
            status = review.get("textualRating") or review.get("title") or "Unverified"
            explanation = review.get("title") or review.get("textualRating") or review.get("explanation") or ""
            url = review.get("url") or ""
            result = {
                "claim": claim,
                "domain": domain_key,
                "status": status,
                "confidence": 0.95,
                "explanation": explanation,
                "sources": [url] if url else []
            }
            return result
    except Exception:
        # proceed to next step
        pass

    # 2) News/Web search + Gemini
    sources = web_verify(claim)
    if sources:
        prompt = f"""
You are an AI misinformation checker.
Claim: "{claim}"
Domain: {domain_key}
Use these sources: {sources}.
Verify if the claim is True/False/Misleading/Unverified.
Respond ONLY in JSON with the following keys:
- claim (string)
- domain (string)
- status (True/False/Misleading/Unverified)
- confidence (float between 0 and 1)
- explanation (string)
- sources (list of URLs)
Return strictly valid JSON only.
"""
        text, err = _generate_with_model(prompt)
        if text:
            parsed = _extract_json_from_text(text)
            if isinstance(parsed, dict):
                # sanitize and normalize
                try:
                    parsed["confidence"] = float(parsed.get("confidence", 0) or 0)
                except Exception:
                    parsed["confidence"] = 0.0
                if "sources" not in parsed or not parsed.get("sources"):
                    parsed["sources"] = sources
                parsed.setdefault("claim", claim)
                parsed.setdefault("domain", domain_key)
                parsed.setdefault("status", "Unverified")
                parsed.setdefault("explanation", parsed.get("explanation", "No explanation provided"))
                return parsed
        # If model failed or returned unparsable text, return measured fallback
        return {
            "claim": claim,
            "domain": domain_key,
            "status": "Unverified",
            "confidence": 0.40,
            "explanation": "Insufficient verifiable evidence from news + model response unparsable.",
            "sources": sources
        }

    # 3) Fallback to trusted static sources + Gemini
    fallback_sources = DOMAIN_SOURCES.get(domain_key, [])
    prompt = f"""
You are an AI misinformation checker.
Domain: {domain_key}
Trusted sources: {', '.join(fallback_sources)}.
Claim: "{claim}"
Return JSON with the keys: claim, domain, status, confidence, explanation, sources (list).
If unsure, set status to "Unverified" and confidence between 0 and 1.
"""
    text, err = _generate_with_model(prompt)
    if text:
        parsed = _extract_json_from_text(text)
        if isinstance(parsed, dict):
            try:
                parsed["confidence"] = float(parsed.get("confidence", 0) or 0)
            except Exception:
                parsed["confidence"] = 0.0
            if "sources" not in parsed or not parsed.get("sources"):
                parsed["sources"] = fallback_sources
            parsed.setdefault("claim", claim)
            parsed.setdefault("domain", domain_key)
            parsed.setdefault("status", "Unverified")
            parsed.setdefault("explanation", parsed.get("explanation", "No explanation provided"))
            return parsed

    # Last fallback: no data
    return {
        "claim": claim,
        "domain": domain_key,
        "status": "Unverified",
        "confidence": 0.30,
        "explanation": "No fact-check / news results and model could not provide a parseable response.",
        "sources": fallback_sources
    }

# ----------------------
# 🌐 Streamlit UI - AI Misinformation Checker
# ----------------------
# ⚠️ NOTE: Your entire UI, HTML, CSS remains untouched below
# ----------------------

import streamlit as st

# ----------------------
# PAGE CONFIGURATION
# ----------------------
st.set_page_config(page_title="AI Misinformation Checker", page_icon="🔎", layout="wide")

# ----------------------
# SESSION STATE (stores UI state across reruns)
# ----------------------
if "show_theme_box" not in st.session_state:
    st.session_state.show_theme_box = False   # controls floating theme window
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"           # default theme

# ----------------------
# THEME BUTTON (Top-right corner)
# ----------------------
col1, col2 = st.columns([8, 1])  
with col2:
    if st.button("🎨 Theme"):  
        st.session_state.show_theme_box = not st.session_state.show_theme_box

# ----------------------
# FLOATING PROMPT
# ----------------------
if st.session_state.show_theme_box:
    st.markdown(
        """
        <style>
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(-10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .theme-prompt {
                position: absolute;
                top: 60px;
                right: 20px;
                background-color: #1E1E1E;
                padding: 12px;
                border-radius: 10px;
                box-shadow: 0px 4px 12px rgba(0,0,0,0.6);
                z-index: 1000;
                animation: fadeIn 0.3s ease-out;
                width: 150px;
                text-align: center;
            }
            .theme-btn {
                display: block;
                margin: 6px auto;
                padding: 8px 14px;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.3s ease;
                border: none;
                width: 100%;
                color: white;
            }
            .dark-btn {
                background: #2a2a2a;
                border: 1px solid #4CAF50;
            }
            .dark-btn:hover {
                background: #4CAF50;
                transform: scale(1.05);
            }
            .light-btn {
                background: #e0e0e0;
                color: black;
                border: 1px solid #36ad7a;
            }
            .light-btn:hover {
                background: #36ad7a;
                color: white;
                transform: scale(1.05);
            }
        </style>
        <div class="theme-prompt">
            <form action="" method="get">
                <button class="theme-btn dark-btn" type="submit" name="theme" value="Dark">🌙 Dark Mode</button>
                <button class="theme-btn light-btn" type="submit" name="theme" value="Light">☀️ Light Mode</button>
            </form>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------
# THEME HANDLING
# ----------------------
query_params = st.query_params
if "theme" in query_params:
    st.session_state.theme = query_params["theme"]
    st.session_state.show_theme_box = False

dark_mode = True if st.session_state.theme == "Dark" else False

# ----------------------
# THEME CSS LOADER
# ----------------------
# ----------------------
# DARK THEME CSS
# ----------------------
def load_dark_theme():
    st.markdown(
        """
        <style>
            body {
                background-color: #0E1117;
                color: #FAFAFA;
            }
            .stApp {
                background-color: #0E1117;
                color: #FAFAFA;
            }
            .stTextArea textarea {
                background-color: #1E1E1E;
                color: #FAFAFA;
                border-radius: 10px;
                border: 1px solid #4CAF50;
                transition: all 0.3s ease-in-out;
            }
            .stTextArea textarea:hover {
                box-shadow: 0px 0px 10px #4CAF50;
                transform: scale(1.01);
            }
            .stSelectbox div[data-baseweb="select"] > div {
                background-color: #1E1E1E;
                color: #FAFAFA;
                border-radius: 10px;
                border: 1px solid #4CAF50;
                transition: all 0.3s ease-in-out;
            }
            .stSelectbox div[data-baseweb="select"] > div:hover {
                box-shadow: 0px 0px 10px #4CAF50;
                transform: scale(1.01);
            }
            .stButton button {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                border: none;
                padding: 8px 16px;
                transition: all 0.3s ease-in-out;
            }
            .stButton button:hover {
                opacity: 0.9;
                transform: scale(1.05);
                box-shadow: 0px 4px 12px #4CAF50;
            }
            h1, h2, h3, h4, h5, h6, p, div {
                color: #FAFAFA;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ----------------------
# LIGHT THEME CSS (FIXED SELECTBOX)
# ----------------------
def load_light_theme():
    st.markdown(
        """
        <style>
            body {
                background-color: #F0F0F2;
                color: #000000;
            }
            .stApp {
                background-color: #F0F0F2;
                color: #000000;
            }

            /* Text Area */
            .stTextArea textarea {
                background-color: #e8e1d1;
                color: #000000;
                border-radius: 10px;
                border: 1px solid #36ad7a;
                transition: all 0.3s ease-in-out;
            }
            .stTextArea textarea:hover {
                box-shadow: 0px 0px 10px #36ad7a;
                transform: scale(1.01);
            }

            /* Collapsed selectbox input */
            .stSelectbox div[data-baseweb="select"] > div {
                background-color: #e8e1d1 !important;
                color: #000000 !important;
                border-radius: 10px;
                border: 1px solid #36ad7a;
                transition: all 0.3s ease-in-out;
            }

            /* Expanded dropdown container */
            ul[role="listbox"] {
                background-color: #e8e1d1 !important;
                border-radius: 10px !important;
                border: 1px solid #36ad7a !important;
                padding: 0;
                margin: 0;
            }

            /* Dropdown options */
            li[role="option"] {
                background-color: #e8e1d1 !important;
                color: #000000 !important;
                padding: 8px 12px;
                cursor: pointer;
            }

            /* Hover effect */
            li[role="option"]:hover {
                background-color: #36ad7a !important;
                color: #ffffff !important;
            }

            /* Selected option */
            li[role="option"][aria-selected="true"] {
                background-color: #36ad7a !important;
                color: #ffffff !important;
            }

            /* Buttons */
            .stButton button {
                background-color: #36ad7a;
                color: white;
                border-radius: 10px;
                border: none;
                padding: 8px 16px;
                transition: all 0.3s ease-in-out;
            }
            .stButton button:hover {
                opacity: 0.9;
                transform: scale(1.05);
                box-shadow: 0px 4px 12px #36ad7a;
            }

            /* Text elements */
            h1, h2, h3, h4, h5, h6, p, div {
                color: #000000;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Load CSS
if dark_mode:
    load_dark_theme()
else:
    load_light_theme()


# ----------------------
# LOGO + TITLE
# ----------------------
if dark_mode:
    st.image("black_logo.png", width=80)
else:
    st.image("white_logo.png", width=80)

st.markdown("<h1 style='margin-top:0'>AI Misinformation Checker</h1>", unsafe_allow_html=True)
st.write("Enter a claim and select a domain to verify if it's true, false, misleading, or unverified.")

# ----------------------
# USER INPUTS
# ----------------------
claim = st.text_area("Enter Claim:")
domain = st.selectbox(
    "Select Domain:", 
    ["Health", "Politics", "Environment", "Technology", "General"]
)


# ----------------------
# VERIFY BUTTON
# ----------------------
if st.button("Verify"):
    if claim.strip():
        with st.spinner("Verifying claim..."):
            result = verify_claim(claim, domain)

        st.subheader("Result")
        st.markdown(f"**Status:** {result['status']}")
        st.markdown(f"**Confidence:** {round(result['confidence']*100, 2)}%")
        st.markdown(f"**Explanation:** {result['explanation']}")
        if result.get("sources"):
            st.markdown("### Sources")
            for src in result["sources"]:
                st.markdown(f"- [{src}]({src})")
    else:
        st.warning("Please enter a claim before verifying.")
