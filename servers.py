from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import urllib.parse
import xml.etree.ElementTree as ET
import requests
import hashlib
import asyncio
import aiohttp
import threading

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Thread-safe Vector DB Setup
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
chroma_client = chromadb.PersistentClient(path="./news_archive_db")
collection = chroma_client.get_or_create_collection(
    name="newspaper_archives_v2", 
    embedding_function=sentence_transformer_ef
)
chroma_lock = threading.Lock() # Prevents SQLite "database locked" crashes during parallel scraping

# 2. AI Client
google_client = genai.Client(api_key="AQ.Ab8RN6IVYpSjqtA9Fqln5uBiElaRM0jnQ8RP8G6Q7Dx_k4y8Gw")

class SearchRequest(BaseModel):
    query: str
    start_year: str = ""
    end_year: str = ""
    page: int = 1

class ChatRequest(BaseModel):
    prompt: str
    context: str = ""
    mode: str = "selected"

@app.get("/")
def serve_index():
    return FileResponse("index.html")

# --- BULLETPROOF ASYNC HELPER ---
def run_async(coro):
    """Safely executes async coroutines from synchronous threads without loop conflicts."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If called from a running async loop, spawn a temp thread to run the sync execution
        result = None
        def run_in_thread():
            nonlocal result
            result = asyncio.run(coro)
        t = threading.Thread(target=run_in_thread)
        t.start()
        t.join()
        return result
    else:
        # Standard synchronous context, safe to run directly
        return asyncio.run(coro)

async def fetch_single_article_text(session, url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=4)) as resp:
            if resp.status == 200:
                html = await resp.text()
                soup = BeautifulSoup(html, 'html.parser')
                paragraphs = [p.text.strip() for p in soup.find_all('p') if len(p.text.strip()) > 50]
                if paragraphs:
                    return " ".join(paragraphs[:5])
    except Exception:
        pass
    return ""

async def enrich_snippets_async(items_list):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_single_article_text(session, item['url']) for item in items_list]
        scraped_texts = await asyncio.gather(*tasks, return_exceptions=True)
        
        for idx, text in enumerate(scraped_texts):
            if isinstance(text, str) and text:
                items_list[idx]['snippet'] = text[:700] + "..."
    return items_list

def fetch_google_news_rss(query: str, start_year: str = "", end_year: str = "", page: int = 1, force_fresh: bool = False):
    try:
        search_term = query.strip()
        
        if force_fresh or (not start_year and not end_year):
            if "when:" not in search_term:
                search_term += " when:2d"
        elif start_year or end_year:
            s_yr = start_year if start_year else "2000"
            e_yr = end_year if end_year else "2026"
            search_term += f" after:{s_yr}-01-01 before:{e_yr}-12-31"

        encoded_query = urllib.parse.quote(search_term)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        resp = requests.get(rss_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        if resp.status_code != 200:
            return []
            
        root = ET.fromstring(resp.content)
        
        raw_items = []
        seen_titles = set()
        items = root.findall('.//item')
        
        start_idx = (page - 1) * 15
        end_idx = page * 15
        
        for item in items[start_idx:end_idx]:
            title = item.find('title').text if item.find('title') is not None else 'News Story'
            
            clean_title = title.split('-')[0].strip().lower()
            if clean_title in seen_titles:
                continue
            seen_titles.add(clean_title)

            source = item.find('source').text if item.find('source') is not None else 'News Outlet'
            link = item.find('link').text if item.find('link') is not None else '#'
            pubDate = item.find('pubDate').text if item.find('pubDate') is not None else "Recent"
            desc = item.find('description').text if item.find('description') is not None else ''
            clean_desc = BeautifulSoup(desc, 'html.parser').text if desc else f"Reporting regarding {title}."

            raw_items.append({
                'title': title,
                'snippet': clean_desc,
                'source': source,
                'date': pubDate[:16],
                'url': link
            })
            if len(raw_items) >= 10:
                break

        # Safely run the async enrichment regardless of calling thread context
        if raw_items:
            raw_items = run_async(enrich_snippets_async(raw_items))
            
        return raw_items
    except Exception as e:
        print(f"RSS Fetch Error: {e}")
        return []

def execute_search(query: str, start_year: str = "", end_year: str = "", page: int = 1, force_fresh: bool = False):
    raw_results = fetch_google_news_rss(query, start_year, end_year, page, force_fresh=force_fresh)

    # DuckDuckGo fallback if RSS fails or returns nothing
    if not raw_results:
        try:
            search_term = f"{query} news"
            if start_year or end_year:
                search_term += f" {start_year} {end_year}".strip()
            with DDGS() as ddgs:
                res = list(ddgs.text(search_term, region="in-en", safesearch="off", max_results=page*10))
                raw_results = res[(page-1)*10 : page*10]
                for r in raw_results:
                    r['snippet'] = r.get('body', r.get('snippet', 'Full reporting available.'))
                    r['source'] = r.get('source', 'News Source')
        except Exception:
            pass

    results, documents, metadatas, ids = [], [], [], []
    for idx, item in enumerate(raw_results):
        title = item.get('title', 'Untitled')
        snippet = item.get('snippet', item.get('body', f'Coverage on {query}.'))
        url = item.get('url', item.get('href', item.get('link', '#')))
        source = item.get('source', 'News Outlet')
        date = item.get('date', "Recent")

        card_data = {
            "id": f"p{page}_{idx}_{hashlib.md5(url.encode()).hexdigest()[:6]}",
            "title": title,
            "snippet": snippet[:220] + "..." if len(snippet) > 220 else snippet,
            "source": source,
            "date": str(date)[:16],
            "url": url
        }
        results.append(card_data)

        doc_id = f"live_{hashlib.md5(url.encode()).hexdigest()}"
        documents.append(f"Title: {title}. Source: {source}. Content: {snippet}")
        metadatas.append({"title": title, "source": source, "year": start_year if start_year else "2026", "url": url})
        ids.append(doc_id)

    # Thread-safe database insertion using upsert to prevent Duplicate ID errors
    if documents:
        with chroma_lock:
            try:
                collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
            except Exception as e:
                print(f"ChromaDB Insert Error: {e}")

    return results

@app.post("/search_news")
def search_news(req: SearchRequest):
    results = execute_search(req.query.strip(), req.start_year.strip(), req.end_year.strip(), req.page)
    return {"status": "success", "articles": results}

@app.post("/search_debate")
async def search_debate(req: SearchRequest):
    query = req.query.strip()
    page = req.page
    loop = asyncio.get_event_loop()

    for_query = f"{query} support"
    against_query = f"{query} opposition"

    for_task = loop.run_in_executor(None, execute_search, for_query, "", "", page, False)
    against_task = loop.run_in_executor(None, execute_search, against_query, "", "", page, False)

    for_articles, against_articles = await asyncio.gather(for_task, against_task)
    return {"status": "success", "for_articles": for_articles, "against_articles": against_articles}

@app.get("/daily_edition")
async def get_daily_edition():
    loop = asyncio.get_event_loop()
    
    # Run in parallel using the ThreadPoolExecutor cleanly
    t1 = loop.run_in_executor(None, execute_search, "Top World Breaking News", "", "", 1, True)
    t2 = loop.run_in_executor(None, execute_search, "India National News Developments", "", "", 1, True)
    t3 = loop.run_in_executor(None, execute_search, "Global Economy Financial Markets", "", "", 1, True)
    t4 = loop.run_in_executor(None, execute_search, "Sports Highlights Scores", "", "", 1, True)
    
    intl, nat, fin, sports = await asyncio.gather(t1, t2, t3, t4)
    
    return {
        "international": intl[:6],
        "national": nat[:6],
        "financial": fin[:6],
        "sports": sports[:6]
    }

@app.get("/market_data")
def get_market_data():
    return {
        "indian": [
            {"symbol": "NIFTY 50", "price": "₹24,512.65", "change": "+1.25%"},
            {"symbol": "RELIANCE", "price": "₹2,950.10", "change": "+0.55%"},
            {"symbol": "TCS", "price": "₹4,120.00", "change": "-0.32%"},
            {"symbol": "HDFC BANK", "price": "₹1,640.50", "change": "+1.80%"},
            {"symbol": "INFY", "price": "₹1,480.20", "change": "-1.10%"}
        ],
        "international": [
            {"symbol": "S&P 500", "price": "$5,204.40", "change": "+0.88%"},
            {"symbol": "NASDAQ", "price": "$16,350.10", "change": "+1.15%"},
            {"symbol": "AAPL", "price": "$175.20", "change": "-0.24%"},
            {"symbol": "MSFT", "price": "$420.55", "change": "+0.45%"},
            {"symbol": "NVDA", "price": "$890.10", "change": "+2.50%"}
        ],
        "crypto": [
            {"symbol": "BTC", "price": "$68,400.00", "change": "+3.45%"},
            {"symbol": "ETH", "price": "$3,550.20", "change": "+2.10%"},
            {"symbol": "SOL", "price": "$154.60", "change": "-1.50%"},
            {"symbol": "BNB", "price": "$592.10", "change": "+0.95%"},
            {"symbol": "XRP", "price": "$0.62", "change": "+4.20%"}
        ]
    }

@app.post("/chat")
def analyze_news(req: ChatRequest):
    selected_context = req.context.strip()
    db_context = ""
    if req.mode == "global":
        # Thread lock not strictly needed for read-only query, but good practice if writes are heavy
        results = collection.query(query_texts=[req.prompt], n_results=8)
        if results['documents'] and results['documents'][0]:
            db_context = "\n\n".join([f"[{m.get('source', 'Archive')}] {m.get('title')}: {d}" 
                                      for d, m in zip(results['documents'][0], results['metadatas'][0])])
    system_prompt = f"""You are an expert, objective news analyst. Adhere strictly to these rules:
1. SOURCE DEPENDENCE: Answer the user's query USING ONLY the context blocks provided below.
2. CITATIONS: ALWAYS cite your sources explicitly in the text.
3. NEUTRALITY MANDATE: If asked for a personal opinion on ANY dispute, reply exactly with: "I cannot provide personal views or opinions on disputes. I can only provide factual data and summarize the articles related to this topic."
---
USER-SELECTED ARTICLES CONTEXT:
{selected_context if selected_context else "No articles dragged into workspace."}
---
GLOBAL ARCHIVE CONTEXT:
{db_context if req.mode == "global" else "Disabled by user."}
"""
    try:
        response = google_client.models.generate_content(
            model="gemma-4-26b-a4b-it",
            contents=f"{system_prompt}\n\nUSER QUESTION: {req.prompt}"
        )
        return {"response": response.text}
    except Exception as e:
        return {"response": f"[API ERROR]: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)