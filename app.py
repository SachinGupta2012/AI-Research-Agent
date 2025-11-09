import os
import streamlit as st
from openai import OpenAI
from ddgs import DDGS
from arxiv import Client, Search
import wikipedia
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("⚠️ OpenAI API key not found. Please add it to your .env file as `OPENAI_API_KEY=your_key_here`.")
    st.stop()

# --- Initialize OpenAI client ---
client = OpenAI(api_key=api_key)

# --- Search functions ---
def search_wikipedia(query):
    try:
        page = wikipedia.page(query)
        return [{
            "source": "Wikipedia",
            "title": page.title,
            "url": page.url,
            "text": page.content[:1000],
        }]
    except Exception:
        return []

def search_duckduckgo(query, max_results=3):
    results = []
    try:
        with DDGS() as ddg:
            for r in ddg.text(query, max_results=max_results):
                results.append({
                    "source": "DuckDuckGo",
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "text": r.get("body", ""),
                })
    except Exception:
        pass
    return results

def search_arxiv(query, max_results=2):
    results = []
    try:
        client_arxiv = Client()
        search = Search(query=query, max_results=max_results)
        for paper in client_arxiv.results(search):
            results.append({
                "source": "arXiv",
                "title": paper.title,
                "url": paper.entry_id,
                "text": paper.summary,
            })
    except Exception:
        pass
    return results

# --- Summarization ---
def summarize_with_gpt(query, docs):
    context = "\n\n".join([f"{d['source']}: {d['title']}\n{d['text']}" for d in docs])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI research assistant that summarizes information clearly and concisely from multiple sources."},
            {"role": "user", "content": f"Question: {query}\n\nSources:\n{context}\n\nSummarize and include source references."},
        ],
    )
    return response.choices[0].message.content.strip()

# --- Streamlit UI ---
st.set_page_config(page_title="AI Research Assistant", page_icon="🧠", layout="centered")

st.title("🧠 AI Research Assistant")
st.write("Ask any research question to get summarized insights from Wikipedia, DuckDuckGo, and arXiv.")

query = st.text_input("Enter your research question:")
if st.button("Search") and query:
    st.info("🔍 Gathering information... Please wait.")
    docs = search_wikipedia(query) + search_duckduckgo(query) + search_arxiv(query)
    
    if not docs:
        st.warning("No information found. Try a different query.")
    else:
        st.success(f"Fetched {len(docs)} documents.")
        with st.spinner("Summarizing with GPT-4o-mini..."):
            answer = summarize_with_gpt(query, docs)
        
        st.markdown("### 🧾 **Answer**")
        st.write(answer)

        st.markdown("### 🔗 **Sources**")
        for i, d in enumerate(docs, 1):
            st.markdown(f"[{i}] **{d['source']}** — [{d['title']}]({d['url']})")

