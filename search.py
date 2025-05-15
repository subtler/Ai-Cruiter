import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import RetrievalQA

load_dotenv()
# ─── CONFIG & SETUP ────────────────────────────────────────────────────────────
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

index_name = "resume-index"

# Load embedding model + vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding_model)

# Load HF LLM
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
llm = HuggingFacePipeline(pipeline=pipe)

# Build retrieval QA chain
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff"
)

# --- Search Prompt ---
print("🔍 AI Resume Search")
while True:
    query = input("\nEnter your search query (or type 'exit'): ")
    if query.lower() == "exit":
        break

    response = qa_chain.invoke({"query": query})

    print(f"\n🧠 Answer:\n{response['result']}\n")
    print("📂 Sources:")
    for doc in response["source_documents"]:
        print(f"- Candidate: {doc.metadata.get('candidate_name')}")
        print(f"  Snippet: {doc.page_content.strip()[:200]}...\n")