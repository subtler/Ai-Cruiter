# create_indexes.py

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
# ─── CONFIG & SETUP ────────────────────────────────────────────────────────────
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

API_KEY    = os.getenv("PINECONE_API_KEY")
DIMENSION  = 384          # for all-MiniLM-L6-v2 embeddings
METRIC     = "cosine"
REGION     = "us-east-1"  # adjust if needed
INDEXES    = ["resume-index", "jd-index"]

# ─── Initialize Pinecone Client ───────────────────────────────────────────────
pc = Pinecone(api_key=API_KEY)

# ─── Ensure Indexes Exist ─────────────────────────────────────────────────────
for name in INDEXES:
    if name in pc.list_indexes():
        print(f"✅ Index '{name}' already exists.")
    else:
        print(f"🛠️  Creating index '{name}'...")
        pc.create_index(
            name=name,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region=REGION)
        )
        print(f"✅ Successfully created '{name}'.")