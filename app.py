import streamlit as st
import google.generativeai as genai
import requests
import os
import json
from dotenv import load_dotenv

# ----------------------
# 🔑 API Key Setup
# ----------------------
# Load environment variables from .env
load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
Fact_check_key = os.getenv("FACTCHECK_API_KEY")

genai.configure(api_key=gemini_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# ----------------------
# 🔗 Trusted Sources
# ----------------------
DOMAIN_SOURCES = {
    "Health": ["WHO", "CDC", "NIH", "ICMR", "PubMed", "MedlinePlus"],
    "Politics": ["Election Commission", "UN", "Reuters", "BBC", "Press Information Bureau India"],
    "Finance": ["RBI", "SEBI", "World Bank", "IMF", "Economic Times"],
    "Climate": ["IPCC", "NASA Climate", "NOAA", "UNEP", "MoEFCC India"],
    "Technology": ["IEEE", "ACM", "Nature Tech", "ScienceDirect"]
}

# ----------------------
# 🔍 Fact Check API
# ----------------------
def fact_check_search(query):
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": Fact_check_key}
    response = requests.get(url, params=params)
    return response.json()

# ----------------------
# ✅ Verify Claim Function
# ----------------------
def verify_claim(claim, domain="Health"):
    fc_result = fact_check_search(claim)

    if "claims" in fc_result and len(fc_result["claims"]) > 0:
        claim_data = fc_result["claims"][0]
        review = claim_data["claimReview"][0]
        return {
            "claim": claim,
            "domain": domain,
            "status": review.get("textualRating", "Unverified"),
            "confidence": 0.95,
            "explanation": review.get("title", "No explanation available"),
            "sources": [review.get("url", "")]
        }

    sources = ", ".join(DOMAIN_SOURCES.get(domain, []))
    prompt = f"""
    You are an AI misinformation checker.

    Task:
    - Verify the following claim in the domain: {domain}.
    - Use ONLY trusted sources: {sources}.
    - If not sure, return status as "Unverified".
    - Always include a confidence score between 0 and 1.

    Format output strictly as JSON with:
    - claim
    - domain
    - status
    - confidence
    - explanation
    - sources (list)
    
    Claim: "{claim}"
    """
    response = model.generate_content(prompt)
    try:
        return json.loads(response.text)
    except:
        return {"claim": claim, "domain": domain, "status": "Unverified", "confidence": 0.3, "explanation": "AI could not parse result", "sources": []}

# ----------------------
# 🌐 Streamlit UI
# ----------------------
st.set_page_config(page_title="AI Misinformation Checker", page_icon="🔎")

st.title("🔎 AI Misinformation Checker")
st.write("Enter a claim and select a domain to verify if it's true, false, misleading, or unverified.")

claim = st.text_area("Enter Claim:")
domain = st.selectbox("Select Domain:", list(DOMAIN_SOURCES.keys()))

if st.button("Verify"):
    if claim.strip():
        with st.spinner("Verifying claim..."):
            result = verify_claim(claim, domain)

        st.subheader("Result")

        st.markdown(f"**Status:** {result.get('status', 'Unverified')}")
        st.markdown(f"**Confidence:** {round(result.get('confidence', 0)*100, 2)}%")
        st.markdown(f"**Explanation:** {result.get('explanation', 'No explanation available')}")

        if result.get("sources"):
            st.markdown("### Sources")
            for src in result["sources"]:
                if src:
                    st.markdown(f"- [{src}]({src})")
    else:
        st.warning("Please enter a claim before verifying.")
