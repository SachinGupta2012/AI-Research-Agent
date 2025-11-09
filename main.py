import os
import wikipedia
import requests
from ddgs import DDGS
import arxiv
from openai import OpenAI
from dotenv import load_dotenv
from arxiv import Client, Search

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def search_wikipedia(query, max_results=2):
    results = []
    try:
        titles = wikipedia.search(query, results=max_results)
        for t in titles:
            page = wikipedia.page(t)
            results.append({"source": "Wikipedia", "title": t, "url": page.url, "text": page.summary})
    except Exception:
        pass
    return results

def search_duckduckgo(query, max_results=3):
    results = []
    with DDGS() as ddg:
        for r in ddg.text(query, max_results=max_results):
            results.append({
                "source": "DuckDuckGo",
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "text": r.get("body", "")
            })
    return results

from arxiv import Client, Search

def search_arxiv(query, max_results=2):
    results = []
    try:
        client = Client()
        search = Search(query=query, max_results=max_results)
        for paper in client.results(search):
            results.append({
                "source": "arXiv",
                "title": paper.title,
                "url": paper.entry_id,
                "text": paper.summary
            })
    except Exception:
        pass
    return results


def summarize_with_gpt(query, docs):
    context = ""
    for i, d in enumerate(docs, start=1):
        context += f"[{i}] ({d['source']}) {d['title']}: {d['text'][:400]}\nURL: {d['url']}\n\n"
    prompt = f"""
You are a helpful AI research assistant.
Use the following sources to answer the user's question clearly and factually.
Cite sources using [numbers].

Question: {query}

Sources:
{context}

Answer:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400
    )
    return response.choices[0].message.content.strip()

def main():
    query = input("Enter your research question: ")
    print("\n🔍 Gathering information...")

    docs = []
    docs += search_wikipedia(query)
    docs += search_duckduckgo(query)
    docs += search_arxiv(query)

    if not docs:
        print("No results found.")
        return

    print(f"Fetched {len(docs)} documents. Summarizing with GPT-4o-mini...\n")
    answer = summarize_with_gpt(query, docs)
    print("===== ANSWER =====")
    print(answer)
    print("\n===== SOURCES =====")
    for i, d in enumerate(docs, start=1):
        print(f"[{i}] {d['source']}: {d['title']} ({d['url']})")

if __name__ == "__main__":
    main()
