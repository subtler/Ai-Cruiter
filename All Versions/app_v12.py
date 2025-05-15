
import os
import tempfile
import streamlit as st
import pandas as pd
from pinecone import Pinecone
from transformers import pipeline
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
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

# Initialize cached resources
embedding_model = get_embedding_model()
resume_store   = get_pinecone_store(RESUME_INDEX)
jd_store       = get_pinecone_store(JD_INDEX)
ner_model      = get_ner_model()

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
    ["Recruiter: Job → Resumes", "Candidate: Resume → Jobs"],
    horizontal=True
)

# ─── RECRUITER MODE: JD → Resumes ───────────────────────────────────────────────
if mode == "Recruiter: Job → Resumes":
    # Job Description Input
    st.header("📄 Job Description Input")
    jd_file = st.file_uploader("Upload JD (.txt)", type=["txt"], key="jd_file")
    if jd_file:
        jd_text = jd_file.read().decode("utf-8")
    else:
        jd_text = st.text_area("Or paste your Job Description here:", height=150)

    if jd_text:
        # Automatic skill extraction
        jd_skills = extract_skills(jd_text)
        st.success("Extracted JD Skills: " + ", ".join(jd_skills))

        # Resume upload & indexing
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

            # Matching logic
            st.header("🤖 Matching Results")
            retr = resume_store.as_retriever(search_kwargs={"filter": {"doc_type": "resume"}})
            jd_vec = embedding_model.embed_query(jd_text)
            docs   = retr.get_relevant_documents(jd_text)

            scored = []
            for doc in docs:
                cos = cosine_similarity([jd_vec], [embedding_model.embed_query(doc.page_content)])[0][0]
                overlap = len(set(doc.metadata.get("skills", [])).intersection(jd_skills))
                score   = 0.7 * cos + 0.3 * (overlap / (len(jd_skills) or 1))
                scored.append((doc, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            top_n = scored[:3]

            st.subheader("Top Matches")
            for i,(doc, score) in enumerate(top_n):
                st.markdown(f"**{i+1}. {doc.metadata['candidate_name']}** — Score: {score:.3f}")
                st.markdown(f"Skills: {', '.join(doc.metadata['skills'])}")
                st.markdown("---")

            st.subheader("All Matches")
            export = []
            for i,(doc, score) in enumerate(scored):
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
                df  = pd.DataFrame(export)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Matches CSV", csv, "matches.csv", "text/csv")

# ─── CANDIDATE MODE: Resume → Jobs ───────────────────────────────────────────────
elif mode == "Candidate: Resume → Jobs":
    st.header("📥 Upload Your Resume")
    resume_u = st.file_uploader("Upload .txt/.pdf resume", type=["txt","pdf"], key="candidate_resume")
    if resume_u:
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + resume_u.name.split('.')[-1]) as tmp:
            tmp.write(resume_u.getvalue())
            path = tmp.name
        loader = TextLoader(path) if resume_u.name.endswith('.txt') else PyPDFLoader(path)
        text = loader.load()[0].page_content

        retr = jd_store.as_retriever(search_kwargs={"filter": {"doc_type": "jd"}})
        jd_docs = retr.get_relevant_documents(text)

        st.subheader("Top Matching Job Descriptions")
        for i,doc in enumerate(jd_docs[:5]):
            name = doc.metadata.get('jd_name', 'Unnamed JD')
            score = getattr(doc, 'score', None)
            st.markdown(f"**{i+1}. {name}** — Score: {score if score is not None else 'N/A'}")
            with st.expander("Preview JD"):
                st.write(doc.page_content.strip())
