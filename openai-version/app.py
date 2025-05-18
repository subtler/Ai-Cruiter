import os
from dotenv import load_dotenv
import streamlit as st
st.set_page_config(page_title="Ai-Cruiter", page_icon="🤖")

# Document loaders and text splitter for ingestion
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Pinecone & LangChain imports
from pinecone import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory

# 1. Load environment variables
load_dotenv()

OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY     = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_RESUME         = os.getenv("PINECONE_INDEX_RESUME")
INDEX_JD             = os.getenv("PINECONE_INDEX_JD")

# 2. Validate env
if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, INDEX_RESUME, INDEX_JD]):
    st.error("Missing one or more environment variables. Check your .env file.")
    st.stop()

# 3. Instantiate Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)


# 5. OpenAI setup
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm        = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

# 6. Wrap Pinecone indexes in LangChain vectorstores
resume_index = pc.Index(INDEX_RESUME)
jd_index     = pc.Index(INDEX_JD)

resume_vstore = PineconeVectorStore(index=resume_index, embedding=embeddings)
jd_vstore     = PineconeVectorStore(index=jd_index,     embedding=embeddings)

# 7. Build the conversational QA chain
memory   = ConversationSummaryMemory(llm=llm)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=resume_vstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory
)

 # ---- DOCUMENT UPLOAD & INGESTION ----
st.sidebar.header("Upload & Ingest Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)
target = st.sidebar.selectbox(
    "Select index to ingest into",
    ("Resume Index", "JD Index")
)
if st.sidebar.button("Ingest to Index"):
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one file.")
    else:
        with st.sidebar.spinner("Ingesting documents..."):
            for uploaded_file in uploaded_files:
                if uploaded_file.name.lower().endswith(".pdf"):
                    loader = PyPDFLoader(uploaded_file)
                else:
                    loader = TextLoader(uploaded_file)
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
                # Choose the correct vectorstore
                vstore = resume_vstore if target == "Resume Index" else jd_vstore
                vstore.add_documents(chunks)
            st.sidebar.success("Ingestion complete!")

# ---- STREAMLIT UI ----
st.title("Ai-Cruiter: Resume & JD Assistant 🤖💼")

tabs = st.tabs(["🔍 Semantic Search", "❓ Q&A on Resume"])

# Tab 1: Semantic Search
with tabs[0]:
    mode       = st.radio("Choose search mode:", ("Resume → JD", "JD → Resume"))
    user_input = st.text_area("Paste your text here (resume or JD):", height=200)
    k          = st.slider("Number of results:", 1, 10, 5)

    if st.button("Search Matches"):
        if not user_input.strip():
            st.warning("Please paste text to search.")
        else:
            vstore = jd_vstore if mode == "Resume → JD" else resume_vstore
            results = vstore.similarity_search(user_input, k=k)
            st.write(f"Top {k} matches ({mode}):")
            for doc in results:
                st.markdown(f"**Score:** {doc.score:.4f}")
                st.write(doc.page_content)
                st.write("---")

# Tab 2: Q&A on Resume
with tabs[1]:
    question = st.text_input("Ask a question about your resume:")
    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            answer = qa_chain.run(question)
            st.markdown("**Answer:**")
            st.write(answer)

            # Show top source docs
            sources = resume_vstore.similarity_search(question, k=3)
            if sources:
                st.markdown("---")
                st.markdown("**Source Documents:**")
                for doc in sources:
                    st.write(doc.metadata.get("source", ""))
                    st.write(doc.page_content)
                    st.write("---")

# ---- RESET INDEX CONTENTS ----
st.sidebar.header("Reset Index Contents")
reset_target = st.sidebar.selectbox(
    "Select index to reset",
    ("Resume Index", "JD Index")
)
if st.sidebar.button("Reset Vectors"):
    with st.sidebar.spinner("Resetting vectors..."):
        if reset_target == "Resume Index":
            resume_index.delete(delete_all=True)
        else:
            jd_index.delete(delete_all=True)
    st.sidebar.success(f"{reset_target} contents have been deleted.")

# Sidebar note
st.sidebar.markdown("---")
st.sidebar.info("Ai-Cruiter now uses the Pinecone client & OpenAI API.\nEnsure your .env is correct before running.")
