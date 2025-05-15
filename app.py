import os
import streamlit as st
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import RetrievalQA

# Setup
os.environ["PINECONE_API_KEY"] = "Your_Pinecone_API_Key"
pc = Pinecone()
index_name = "resume-index"

# Embeddings + Vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding_model)

# LLM pipeline
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
llm = HuggingFacePipeline(pipeline=pipe)

# Retrieval chain
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True, chain_type="stuff")

# --- Streamlit UI ---
st.set_page_config(page_title="AI Resume Matcher", page_icon="🧠")
st.title("🧠 AI Resume Matcher with HuggingFace + Pinecone")

query = st.text_input("Enter a skill, tool, or experience to search resumes:")

if query:
    response = qa_chain.invoke({"query": query})
    st.markdown("### 🤖 Suggested Match:")
    st.write(response["result"])

    st.markdown("### 📂 Matched Resumes:")
    for i, doc in enumerate(response["source_documents"]):
        st.markdown(f"**{i+1}. Candidate:** `{doc.metadata.get('candidate_name')}`")
        with st.expander("📄 Snippet"):
            st.write(doc.page_content.strip())