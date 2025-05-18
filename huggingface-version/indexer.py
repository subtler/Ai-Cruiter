import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()
# ─── CONFIG & SETUP ────────────────────────────────────────────────────────────
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

index_name = "resume-index"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_resumes(folder_path="data"):
    all_chunks = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
            name = filename.replace(".pdf", "")
        elif filename.endswith(".txt"):
            loader = TextLoader(path)
            name = filename.replace(".txt", "")
        else:
            continue

        docs = loader.load()
        for doc in docs:
            doc.metadata["candidate_name"] = name

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    return all_chunks

# Load and embed
docs = load_resumes("data")

# Push to Pinecone
vectorstore = PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embedding_model,
    index_name=index_name
)

print(f"✅ Indexed {len(docs)} chunks into Pinecone.")