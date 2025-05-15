
import os
import tempfile
import streamlit as st
import pandas as pd
from pinecone import Pinecone
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity


# ─── CONFIG & SETUP ────────────────────────────────────────────────────────────
os.environ["PINECONE_API_KEY"] = "your-pinecone-api-key"  # ← replace!
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

RESUME_INDEX = "resume-index"
JD_INDEX     = "jd-index"

# ─── CACHING MODELS & STORES ───────────────────────────────────────────────────
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_pinecone_store(index_name):
    return PineconeVectorStore(index_name=index_name, embedding=get_embedding_model())

@st.cache_resource
def get_ner_model():
    return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

@st.cache_resource
def get_llm_chain():
    # LLM setup for chat with memory
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model     = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    pipe      = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
    llm       = HuggingFacePipeline(pipeline=pipe)
    summary_memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )
    retriever = get_pinecone_store(RESUME_INDEX).as_retriever(search_kwargs={"filter": {"doc_type": "resume"}})
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=summary_memory,
        return_source_documents=True,
        output_key="answer"
    )

# Initialize resources
embedding_model = get_embedding_model()
resume_store    = get_pinecone_store(RESUME_INDEX)
jd_store        = get_pinecone_store(JD_INDEX)
ner_model       = get_ner_model()
qa_chain        = get_llm_chain()

# ─── SKILL EXTRACTION ──────────────────────────────────────────────────────────
def extract_skills(text):
    ents = ner_model(text)
    return list({e["word"] for e in ents if e["entity_group"] in ["ORG", "MISC"]})

# ─── STREAMLIT UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Resume Matcher", page_icon="🧠", layout="wide")
st.title("🧠 AI Resume Matcher")

# ─── SIDEBAR: Cleanup Vectors ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧹 Cleanup Tools")
    with st.expander("⚠️ Clear All Stored Vectors"):
        st.warning("This will delete all vectors in both Pinecone indexes.")
        pwd = st.text_input("Admin password:", type="password")
        if st.button("Clear Vectors"):
            if pwd == "123":
                with st.spinner("Clearing vectors..."):
                    for idx_name in [RESUME_INDEX, JD_INDEX]:
                        idx = pc.Index(idx_name)
                        idx.delete(delete_all=True)
                st.success("✅ All vectors cleared.")
            else:
                st.error("❌ Wrong password")

# ─── MODE SELECTION ─────────────────────────────────────────────────────────────
mode = st.radio(
    "Select Mode:",
    ["Recruiter: Job → Resumes", "Chat with Memory"],
    horizontal=True,
    index=0,                          # default to Recruiter mode
    key="mode_selector"
)

# ─── RECRUITER MODE: JD → Resumes ───────────────────────────────────────────────
if mode == "Recruiter: Job → Resumes":
    st.header("📄 Job Description Input")
    jd_file = st.file_uploader("Upload JD (.txt)", type=["txt"], key="jd_file")
    if jd_file:
        jd_text = jd_file.read().decode("utf-8")
    else:
        jd_text = st.text_area("Or paste your Job Description here:", height=150)

    if jd_text:
        jd_skills = extract_skills(jd_text)
        st.success("Extracted JD Skills: " + ", ".join(jd_skills))

        st.header("📥 Upload & Index Resumes")
        resumes = st.file_uploader(
            "Upload resumes (.txt/.pdf)", type=["txt","pdf"], accept_multiple_files=True, key="resumes"
        )
        if resumes:
            for f in resumes:
                with tempfile.NamedTemporaryFile(delete=False, suffix="." + f.name.split('.')[-1]) as tmp:
                    tmp.write(f.getvalue())
                    path = tmp.name
                loader = TextLoader(path) if f.name.endswith('.txt') else PyPDFLoader(path)
                docs = loader.load()
                candidate = f.name.rsplit('.',1)[0]
                for doc in docs:
                    skills = extract_skills(doc.page_content[:1000])
                    doc.metadata.update({
                        "candidate_name": candidate,
                        "doc_type": "resume",
                        "source_file": f.name,
                        "skills": skills
                    })
                chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(docs)
                PineconeVectorStore.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    index_name=RESUME_INDEX
                )
            st.success(f"Indexed {len(resumes)} resume(s).")

            # Matching logic with deduplication
            st.header("🤖 Matching Results")
            retr = resume_store.as_retriever(search_kwargs={"filter": {"doc_type": "resume"}})
            jd_vec = embedding_model.embed_query(jd_text)
            docs = retr.get_relevant_documents(jd_text)

            # Score each document chunk
            scored = []
            for doc in docs:
                cos = cosine_similarity([jd_vec], [embedding_model.embed_query(doc.page_content)])[0][0]
                overlap = len(set(doc.metadata.get("skills", [])).intersection(jd_skills))
                score = 0.7 * cos + 0.3 * (overlap / (len(jd_skills) or 1))
                scored.append((doc, score))

            # Deduplicate by candidate: pick highest score per candidate
            candidate_map = {}
            for doc, score in scored:
                name = doc.metadata['candidate_name']
                if name not in candidate_map or score > candidate_map[name][1]:
                    candidate_map[name] = (doc, score)
            unique_scored = list(candidate_map.values())
            unique_scored.sort(key=lambda x: x[1], reverse=True)

            # Top 3
            top_n = unique_scored[:3]
            st.subheader("Top Matches")
            for i, (doc, score) in enumerate(top_n):
                st.markdown(f"**{i+1}. {doc.metadata['candidate_name']}** — Score: {score:.3f}")
                st.markdown(f"Skills: {', '.join(doc.metadata['skills'])}")
                st.markdown("---")

            # All matches
            st.subheader("All Matches")
            export = []
            for i, (doc, score) in enumerate(unique_scored):
                skills = doc.metadata.get("skills", [])
                st.markdown(f"**{i+1}. {doc.metadata['candidate_name']}** — Score: {score:.3f}")
                with st.expander("Preview Resume Snippet"):
                    st.write(doc.page_content[:300] + "...")
                export.append({
                    "Rank": i+1,
                    "Candidate": doc.metadata['candidate_name'],
                    "Skills": ",".join(skills),
                    "Score": f"{score:.3f}"
                })
            if export:
                df = pd.DataFrame(export)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Matches CSV", csv, "matches.csv", "text/csv")

# ─── CHAT WITH MEMORY MODE ─────────────────────────────────────────────────────
else:
    st.header("🗨️ Chat with Memory")
    if st.button("Clear Chat History"):
        qa_chain.memory.clear()
        st.success("Chat history cleared!")
    query = st.text_input("Ask a follow-up question:")
    if query:
        res = qa_chain.invoke({"question": query})
        st.markdown("**Answer:**")
        st.write(res["answer"])
        if res.get("source_documents"):
            st.markdown("**Sources:**")
            for i, doc in enumerate(res["source_documents"]):
                name = doc.metadata.get("candidate_name", doc.metadata.get("jd_name", ""))
                st.markdown(f"{i+1}. {name}")
                with st.expander("Preview"):
                    st.write(doc.page_content[:200] + "...")
    st.markdown("### Conversation Summary")
    st.write(qa_chain.memory.buffer)
    st.markdown("### Full Chat History")
    for msg in qa_chain.memory.chat_memory.messages:
        role = "You" if msg.type == "human" else "Bot"
        st.markdown(f"**{role}:** {msg.content}")
