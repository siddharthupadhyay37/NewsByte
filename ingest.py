import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import glob
import os

print("⚡ Loading embedding model...")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./news_archive_db")
collection = client.get_or_create_collection(
    name="newspaper_archives_v2",
    embedding_function=sentence_transformer_ef
)

# Explicitly list your files, including the renamed one
csv_files = ["news_articles.csv"] + [f for f in glob.glob("articles*.csv") if f != "news_articles.csv"]

# Filter out files that don't actually exist in the folder
csv_files = [f for f in csv_files if os.path.exists(f)]

if not csv_files:
    print("❌ Error: No news CSV files found in the folder!")
else:
    print(f"📖 Found dataset files to ingest: {csv_files}")

    global_idx = 0
    BATCH_SIZE = 100

    for file_path in csv_files:
        print(f"\n📂 Processing {file_path}...")
        df = pd.read_csv(file_path)
        
        content_col = next((col for col in ["content", "text", "article"] if col in df.columns), None)
        if not content_col:
            print(f"⚠️ Skipping {file_path}: Could not find content column.")
            continue
            
        df = df.dropna(subset=[content_col]).reset_index(drop=True)
        total_rows = len(df)
        print(f"🚀 Ingesting {total_rows} articles from {file_path}...")

        for i in range(0, total_rows, BATCH_SIZE):
            batch = df.iloc[i:i + BATCH_SIZE]
            
            documents = []
            metadatas = []
            ids = []
            
            for idx, row in batch.iterrows():
                text_content = str(row[content_col])[:2000]
                source = str(row.get("publication", row.get("source", "Kaggle Archive")))
                year = str(row.get("year", str(row.get("date", "Unknown"))[:4]))
                title = str(row.get("title", "Untitled Article"))
                
                documents.append(text_content)
                metadatas.append({"source": source, "year": year, "title": title})
                ids.append(f"doc_{global_idx}")
                global_idx += 1
            
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ Progress: Ingested batch up to article {i + len(batch)} / {total_rows} in {file_path}")

    print("\n🎉 Full dataset ingestion complete! Ready to load up your cosmic terminal.")