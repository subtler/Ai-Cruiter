# create_indexes.py

import os
from pinecone import Pinecone, ServerlessSpec

# ─── Configuration ────────────────────────────────────────────────────────────
# You can also set the API key as an environment variable externally
os.environ["PINECONE_API_KEY"] = "pcsk_6ANMxB_NBF6TZziCKrn6kWNDskfdQzUj5GU7AJYtFWkWwsRefuXBdrJxRSxrvRe1Y2Nbi2"

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