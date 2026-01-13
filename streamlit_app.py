# streamlit_app.py

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from groq import Groq
import logging

# Setup
logging.basicConfig(level=logging.INFO)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# LLM Wrapper

def call_llm(prompt: str, model: str = "mixtral-8x7b-32768") -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional fund formation analyst, compliance officer, and LP relations manager."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# Functional Modules

def draft_fund_documents(text: str) -> str:
    prompt = f"Draft private fund documents (LPA, PPM, sub doc outlines) for a fund with: {text}"
    return call_llm(prompt)

def build_fund_model(df: pd.DataFrame) -> str:
    prompt = f"Create IRR, MOIC, and waterfall model summary for: {df.head(3).to_dict()}"
    return call_llm(prompt)

def generate_lp_comms(df: pd.DataFrame) -> str:
    prompt = f"Write a professional LP update letter based on: {df.head(3).to_dict()}"
    return call_llm(prompt)

def generate_capital_call(df: pd.DataFrame) -> str:
    prompt = f"Generate a capital call notice for LPs using: {df.head(3).to_dict()}"
    return call_llm(prompt)

def compliance_monitoring(text: str) -> str:
    prompt = f"Analyze this fund text for compliance red flags and ESG alignment: {text}"
    return call_llm(prompt)

def fund_ops_assistant(question: str, context: str) -> str:
    prompt = f"Fund context: {context}\nUser asks: {question}\nProvide detailed response as an expert fund ops AI."
    return call_llm(prompt)

# UI

def main():
    st.set_page_config("FundFoundry AI", page_icon="🏗", layout="wide")
    st.title("🏗 FundFoundry AI – Institutional Fund Builder & Compliance Copilot")
    st.write("Form, model, and run your fund like a pro — instantly powered by LLMs.")

    uploaded_file = st.file_uploader("Upload fund data CSV (for modeling, notices, etc.)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Data uploaded.")
    else:
        df = pd.DataFrame()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📄 Fund Docs",
        "📊 Fund Model",
        "📨 LP Comms",
        "💵 Capital Call",
        "🛡 Compliance",
        "💬 Fund Assistant"
    ])

    with tab1:
        st.subheader("📄 Draft Fund Formation Docs")
        text = st.text_area("Describe your fund structure or goals")
        if st.button("Generate Docs"):
            if not text:
                st.error("Enter fund description.")
            else:
                out = draft_fund_documents(text)
                st.text_area("Generated Docs", value=out, height=400)

    with tab2:
        st.subheader("📊 Fund Model Summary")
        if st.button("Build Model"):
            if df.empty:
                st.error("Upload fund data.")
            else:
                model = build_fund_model(df)
                st.text_area("Model Output", value=model, height=400)

    with tab3:
        st.subheader("📨 LP Update Letter")
        if st.button("Write LP Update"):
            if df.empty:
                st.error("Upload fund data.")
            else:
                update = generate_lp_comms(df)
                st.text_area("LP Update", value=update, height=400)

    with tab4:
        st.subheader("💵 Capital Call Notice")
        if st.button("Generate Call"):
            if df.empty:
                st.error("Upload capital data.")
            else:
                notice = generate_capital_call(df)
                st.text_area("Capital Call", value=notice, height=300)

    with tab5:
        st.subheader("🛡 Compliance/ESG Review")
        doc = st.text_area("Paste fund legal or marketing text")
        if st.button("Check Compliance"):
            if not doc:
                st.error("Paste your text.")
            else:
                flags = compliance_monitoring(doc)
                st.text_area("Compliance Review", value=flags, height=400)

    with tab6:
        st.subheader("💬 Fund Operations Assistant")
        context = st.text_area("Paste fund background")
        q = st.text_input("Ask a fund question")
        if st.button("Ask AI"):
            if not context or not q:
                st.error("Fill both fields.")
            else:
                answer = fund_ops_assistant(q, context)
                st.markdown(f"**AI Answer:** {answer}")

if __name__ == "__main__":
    main()
