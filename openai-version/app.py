import os
import json
import statistics
from dotenv import load_dotenv
import streamlit as st
# --- Load persisted resume metadata from disk ---
METADATA_PATH = "resume_metadata.json"
try:
    with open(METADATA_PATH, "r") as f:
        st.session_state["resume_metadata"] = json.load(f)
except FileNotFoundError:
    st.session_state["resume_metadata"] = {}
import tempfile
import pandas as pd
from pinecone.openapi_support.exceptions import PineconeApiException
import re
from difflib import SequenceMatcher

# Initialize shortlist storage
if 'shortlists' not in st.session_state:
    st.session_state['shortlists'] = {}

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
# Additional imports for prompt/chain
from langchain import PromptTemplate
from langchain.chains import LLMChain

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

# Cached function for generating roles and JD from resume
@st.cache_data
def generate_roles_and_jd(resume_text: str) -> str:
    prompt = PromptTemplate(
        input_variables=["resume_text"],
        template="""
You are a career advisor. Given this resume:

{resume_text}

1. Generate a concise job description for a suitable role.
2. List 5 possible job titles the candidate can apply for.
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(resume_text=resume_text)

@st.cache_data
def extract_resume_metadata(resume_text: str) -> dict:
    # Use LLM to extract structured metadata as JSON
    metadata_prompt = PromptTemplate(
        input_variables=["resume_text"],
        template="""
Extract JSON metadata from this resume. Return the following keys:
- skills: a list of skill names
- tools: a list of tool names
- experience_years: total years of experience as a number
- certifications: list of certifications
- education: list of degrees/institutions

Resume:
{resume_text}
"""
    )
    chain = LLMChain(llm=llm, prompt=metadata_prompt)
    response = chain.run(resume_text=resume_text)
    metadata = json.loads(response)
    # Ensure metadata fields are valid Pinecone types
    # Flatten education entries if they are dicts
    edu = metadata.get("education", [])
    if isinstance(edu, list) and edu and isinstance(edu[0], dict):
        # Extract just the degree or institution names
        metadata["education"] = [entry.get("degree", "") for entry in edu if isinstance(entry, dict)]
    # Convert any non-list metadata into a simple list or string
    for key, val in list(metadata.items()):
        if isinstance(val, list):
            # ensure all items are strings
            metadata[key] = [str(item) for item in val]
        elif not isinstance(val, (str, int, float, bool)):
            # fallback to string
            metadata[key] = str(val)
    return metadata

# Fuzzy match helper
def fuzzy_match_list(jd_list, res_list, threshold=0.8):
    """Return list of resume items that fuzzy-match any JD item above threshold."""
    matches = set()
    jd_norm = [j.lower().strip() for j in jd_list]
    res_norm = [r.lower().strip() for r in res_list]
    for jd in jd_norm:
        for res in res_norm:
            # exact substring or fuzzy ratio
            ratio = SequenceMatcher(None, jd, res).ratio()
            if jd in res or res in jd or ratio >= threshold:
                matches.add(res)
    return list(matches)

# 6. Wrap Pinecone indexes in LangChain vectorstores
resume_index = pc.Index(INDEX_RESUME)
jd_index     = pc.Index(INDEX_JD)

resume_vstore = PineconeVectorStore(index=resume_index, embedding=embeddings)
jd_vstore     = PineconeVectorStore(index=jd_index,     embedding=embeddings)

# 7. Build the conversational QA chain
memory   = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=resume_vstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    return_source_documents=True,
    verbose=True
)

#
# ---- STREAMLIT UI ----
st.title("Ai-Cruiter: Resume & JD Assistant 🤖💼")

# Main Modes
tabs = st.tabs(["🧑‍💼 Recruiter Mode", "👤 Candidate Mode", "📚 Library"])

with tabs[0]:
    st.header("Recruiter Mode")


    # (Removed Uploaded Resumes section and separator)

    jd_input = st.text_area("Paste a Job Description (JD):", height=200)
    if st.button("Find Matching Resumes"):
        if not jd_input.strip():
            st.warning("Please paste a Job Description to search.")
        else:
            # Ensure resumes have been ingested
            resume_meta = st.session_state.get("resume_metadata", {})
            if not resume_meta:
                st.error("No resumes have been ingested yet. Please go to the Admin tab and ingest resumes first.")
                st.stop()

            # Retrieve semantic (cosine) scores from Pinecone
            raw_semantic = resume_vstore.similarity_search_with_score(jd_input, k=50)
            # Group by source and track best semantic score per resume
            sem_scores = {}
            for doc, score in raw_semantic:
                src = doc.metadata.get("source_file", "Unknown")
                sem_scores[src] = max(score, sem_scores.get(src, 0))

            # Extract JD metadata for meta scoring
            jd_meta = extract_resume_metadata(jd_input)
            jd_skills = set(jd_meta.get("skills", []))
            jd_tools  = set(jd_meta.get("tools", []))
            try:
                jd_exp_val = float(jd_meta.get("experience_years", 0))
            except:
                jd_exp_val = 0.0

            w_skill, w_tool, w_exp = 0.5, 0.3, 0.2  # weights inside the 30% meta portion
            w_semantic, w_meta = 0.7, 0.3

            # Build combined scores
            data = []
            for src, meta in resume_meta.items():
                # Semantic component
                sem_score = sem_scores.get(src, 0.0)

                # Metadata component
                res_skills = set(meta.get("skills", []))
                res_tools  = set(meta.get("tools", []))
                try:
                    res_exp_val = float(meta.get("experience_years", 0))
                except:
                    res_exp_val = 0.0

                matched_skills = fuzzy_match_list(jd_meta.get("skills", []), meta.get("skills", []))
                matched_tools  = fuzzy_match_list(jd_meta.get("tools", []),  meta.get("tools", []))
                skill_score = len(matched_skills) / (len(jd_skills) or 1)
                tool_score  = len(matched_tools)  / (len(jd_tools)  or 1)
                exp_score   = min(res_exp_val / jd_exp_val, 1.0) if jd_exp_val>0 else 1.0

                meta_score = w_skill*skill_score + w_tool*tool_score + w_exp*exp_score

                # Combined score
                total_score = w_semantic*sem_score + w_meta*meta_score

                data.append({
                    "Source": src,
                    "Score": round(total_score, 4),
                    "Skills Matched": ", ".join(matched_skills),
                    "Tools Matched":  ", ".join(matched_tools),
                    "Experience Match": f"{res_exp_val} yrs"
                })

            # Display sorted table
            df = pd.DataFrame(sorted(data, key=lambda x: x["Score"], reverse=True))
            st.dataframe(df)

# Candidate Mode: generate JD & roles from resume
from langchain import PromptTemplate
from langchain.chains import LLMChain

with tabs[1]:
    st.header("Candidate Mode")
    candidate_file = st.file_uploader("Upload your resume (PDF or TXT):", type=["pdf", "txt"], key="candidate_resume")
    if st.button("Generate Roles & JD"):
        if not candidate_file:
            st.warning("Please upload your resume.")
        else:
            # Write UploadedFile to a temp file
            suffix = os.path.splitext(candidate_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(candidate_file.read())
                temp_path = tmp.name

            # Load and combine text from the temp file
            if candidate_file.name.lower().endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
            else:
                loader = TextLoader(temp_path)
            docs = loader.load()
            full_text = "\n".join([d.page_content for d in docs])

            # Clean up the temp file
            os.remove(temp_path)

            # Generate using cached function
            output = generate_roles_and_jd(full_text)

            # Robustly extract job description and roles
            # Attempt to capture between 'Job Description:' and 'Possible Job Titles:'
            desc_match = re.search(r"Job Description:\s*(.*?)\s*Possible Job Titles:", output, re.S)
            if desc_match:
                job_desc = desc_match.group(1).strip()
                roles_section = output.split("Possible Job Titles:")[-1]
            else:
                # Fallback: first line after the colon
                lines = [l.strip() for l in output.split("\n") if l.strip()]
                if lines and ":" in lines[0]:
                    job_desc = lines[0].split(":",1)[1].strip()
                    roles_section = "\n".join(lines[1:])
                else:
                    job_desc = output.strip()
                    roles_section = ""

            # Extract roles as numbered entries, skipping empty entries
            roles = []
            for line in roles_section.split("\n"):
                m = re.match(r"\s*\d+\.\s*(.*)", line)
                if m:
                    role = m.group(1).strip()
                    if role:  # only append non-empty role text
                        roles.append(role)

            # Remove any stray numeric-only lines (e.g., "2.")
            job_desc = "\n".join(
                line for line in job_desc.split("\n")
                if not re.match(r"^\s*\d+\.\s*$", line)
            )
            # Display in two-column split
            cols = st.columns([2, 1])
            with cols[0]:
                st.subheader("📋 Job Description")
                st.write(job_desc)
            with cols[1]:
                st.subheader("🎯 Suggested Roles")
                for role in roles:
                    st.write(f"- {role}")

            # Button to reset for a new resume
            if st.button("New Resume", key="new_resume"):
                st.session_state["candidate_resume"] = None
                st.experimental_rerun()



# Chain to extract targeted bullet points from resume passages
extract_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["passages", "topic"],
        template="""
You are an expert at scanning resumes. Given the following resume passages:

{passages}

Extract bullet points specifically about the candidate's experience with {topic}.  
Return each bullet prefixed with a dash.
"""
    )
)

# Library Tab: Uploaded Resumes
with tabs[2]:
    st.header("Library: Uploaded Resumes")
    resume_meta = st.session_state.get("resume_metadata", {})
    count = len(resume_meta)
    st.write(f"Total resumes uploaded: **{count}**")
    if count > 0:
        st.write("**Resume Files:**")
        for source in list(resume_meta.keys()):
            cols = st.columns([5, 1])
            with cols[0]:
                st.write(f"- {source}")
            with cols[1]:
                # Delete button for each resume
                if st.button("Delete", key=f"del_{source}"):
                    try:
                        # Delete all vectors with this source_file metadata
                        resume_index.delete(delete_all=True, filter={"source_file": source})
                        # Remove from session state and disk
                        st.session_state["resume_metadata"].pop(source, None)
                        with open(METADATA_PATH, "w") as f:
                            json.dump(st.session_state["resume_metadata"], f)
                        st.success(f"Removed resume '{source}' and its vectors.")
                    except PineconeApiException as e:
                        st.error(f"Error deleting resume '{source}': {e}")
    else:
        st.info("No resumes uploaded yet. Please ingest in the sidebar.")


# ---- SIDEBAR: Admin ----
st.sidebar.header("Admin Panel")
admin_uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key="admin_docs_upload"
)
index_choice = st.sidebar.selectbox(
    "Select index to ingest into",
    (INDEX_RESUME, INDEX_JD),
    label_visibility="visible"
)
if st.sidebar.button("Ingest to Index"):
    if not admin_uploaded_files:
        st.sidebar.warning("Please upload at least one file.")
    else:
        with st.spinner("Ingesting documents..."):
            total_chunks = 0
            for uploaded_file in admin_uploaded_files:
                # Write streamlit UploadedFile to a temporary file
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name

                # Use the temp file path for the loader
                if uploaded_file.name.lower().endswith(".pdf"):
                    loader = PyPDFLoader(temp_path)
                else:
                    loader = TextLoader(temp_path)

                docs = loader.load()
                full_text = "\n".join([d.page_content for d in docs])
                # Extract structured metadata once per resume
                metadata = extract_resume_metadata(full_text)
                metadata['source_file'] = uploaded_file.name
                # Store resume-wide metadata in session_state for recruiter mode
                if 'resume_metadata' not in st.session_state:
                    st.session_state['resume_metadata'] = {}
                st.session_state['resume_metadata'][uploaded_file.name] = metadata
                # Persist updated metadata to disk
                with open(METADATA_PATH, "w") as f:
                    json.dump(st.session_state["resume_metadata"], f)
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
                total_chunks += len(chunks)
                # Attach metadata to each chunk
                for chunk in chunks:
                    if chunk.metadata is None:
                        chunk.metadata = {}
                    chunk.metadata.update(metadata)
                vstore = resume_vstore if index_choice == INDEX_RESUME else jd_vstore
                try:
                    vstore.add_documents(chunks)
                except PineconeApiException as e:
                    st.sidebar.error(f"Failed to ingest vectors: {e}. This usually means your index dimension ({INDEX_RESUME} or {INDEX_JD}) doesn't match 1536. Please create a new Pinecone index with dimension=1536 and update your .env accordingly.")
                    continue  # Skip this file and continue with the next one

                # Clean up the temporary file
                os.remove(temp_path)
            # Display ingestion summary
            st.sidebar.info(f"Ingested a total of {total_chunks} document chunks into the '{index_choice}' index.")
            # Show index stats
            stats = (resume_index if index_choice == INDEX_RESUME else jd_index).describe_index_stats()
            st.sidebar.write("Current index stats:", stats)
        st.sidebar.success("Ingestion complete!")

st.sidebar.markdown("---")
st.sidebar.subheader("Reset Index Contents")
reset_target = st.sidebar.selectbox("Select index to reset", ("Resume Index", "JD Index"), key="reset_choice")
if st.sidebar.button("Reset Vectors"):
    with st.spinner("Resetting vectors..."):
        if reset_target == "Resume Index":
            resume_index.delete(delete_all=True)
            # Also clear persisted metadata if resetting the resume index
            st.session_state["resume_metadata"] = {}
            try:
                os.remove(METADATA_PATH)
            except FileNotFoundError:
                pass
        else:
            jd_index.delete(delete_all=True)
    st.sidebar.success(f"{reset_target} contents have been deleted.")
