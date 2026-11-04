import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from google import genai # NEW: Google's Cloud API SDK

# Initialize the robust downloader (Sentence Transformers)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Initialize the Local Vector Database
client = chromadb.PersistentClient(path="./news_archive_db")
collection = client.get_or_create_collection(name="newspaper_archives_v2", embedding_function=sentence_transformer_ef)

# Load Mock News Archives
if collection.count() == 0:
    collection.add(
        documents=[
            "2005 Tech News: The adoption of basic broadband has grown by 40% this year. Smartphones are practically non-existent.",
            "2015 Tech News: 4G networks have become the global standard, and mobile internet usage has officially surpassed desktop usage.",
            "2025 Tech News: AI integration and 6G testing have fundamentally changed tech. Autonomous agent networks manage 30% of web traffic."
        ],
        metadatas=[{"source": "NYT Archive", "year": "2005"}, {"source": "WSJ Archive", "year": "2015"}, {"source": "TechCrunch", "year": "2025"}],
        ids=["doc1", "doc2", "doc3"]
    )

st.set_page_config(page_title="News Archive AI", layout="centered")
st.title("📰 News Archive AI (Powered by Cloud Gemma 4)")

# Paste your actual Google AI Studio API key here!
google_client = genai.Client(api_key="AQ.Ab8RN6IVYpSjqtA9Fqln5uBiElaRM0jnQ8RP8G6Q7Dx_k4y8Gw")

tab1, tab2 = st.tabs(["Search Archives (Feed)", "Analyze & Compare (Chatbot)"])

# --- TAB 1: Inshorts-Style Search Feed ---
with tab1:
    st.subheader("Topic Search")
    search_query = st.text_input("Enter a topic (e.g., 'tech growth')")
    
    if st.button("Search Archives"):
        results = collection.query(query_texts=[search_query], n_results=3)
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            st.info(f"**{meta['source']} ({meta['year']})**\n\n{doc}")

# --- TAB 2: Conversational Analysis (Now using Cloud API) ---
with tab2:
    st.subheader("AI Analysis")
    user_prompt = st.text_area("Ask a complex question (e.g., 'Compare tech growth from 2005 to 2025')")
    
    if st.button("Generate Analysis"):
        results = collection.query(query_texts=[user_prompt], n_results=3)
        context = "\n\n".join(results['documents'][0])
        
        system_prompt = f"""You are an expert historian and news analyst. You must answer the user's question USING ONLY the provided news archive text below. If the answer cannot be found in the text, you must explicitly say 'I cannot answer this based on the available archives.' Do not use outside knowledge.
        
ARCHIVE TEXT:
{context}"""
        
        with st.spinner("Cloud Gemma 4 is analyzing the archives..."):
            try:
                # Send the request to Google's Enterprise Cloud Servers instead of local Ollama
                response = google_client.models.generate_content(
                    model="gemma-4-26b-a4b-it", 
                    contents=f"{system_prompt}\n\nUSER QUESTION: {user_prompt}"
                )
                
                st.success(response.text)
                
                with st.expander("View Cited Sources"):
                    for meta in results['metadatas'][0]:
                        st.write(f"- {meta['source']} ({meta['year']})")
                        
            except Exception as e:
                st.error(f"API Error: Make sure your API key is correct! Details: {e}")