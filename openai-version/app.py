import os
from dotenv import load_dotenv
import streamlit as st
import json
import tempfile
import pandas as pd
from pinecone.openapi_support.exceptions import PineconeApiException

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

#
# ---- STREAMLIT UI ----
st.title("Ai-Cruiter: Resume & JD Assistant 🤖💼")

# Main Modes
tabs = st.tabs(["🧑‍💼 Recruiter Mode", "👤 Candidate Mode", "🔧 Admin"])

with tabs[0]:
    st.header("Recruiter Mode")

    # Search naming and loading
    search_name = st.text_input("Search Name:", key="search_name")
    saved_searches = list(st.session_state['shortlists'].keys())
    load_option = st.selectbox("Load Saved Search", [""] + saved_searches, key="load_option")
    if load_option:
        selected = st.session_state['shortlists'][load_option]
        st.write(f"Shortlisted candidates for '{load_option}':")
        st.write(selected)

    jd_input = st.text_area("Paste a Job Description (JD):", height=200)
    if st.button("Find Matching Resumes", key="find_resumes"):
        if not jd_input.strip():
            st.warning("Please paste a Job Description to search.")
        else:
            with st.spinner("Searching resumes..."):
                raw_results = resume_vstore.similarity_search_with_score(jd_input, k=50)
                # Group by source_file
                grouped = {}
                for doc, score in raw_results:
                    source = doc.metadata.get('source_file', 'Unknown')
                    if source not in grouped:
                        grouped[source] = {"best_score": score, "chunks": [(doc, score)]}
                    else:
                        grouped[source]["chunks"].append((doc, score))
                        # Update best score if this chunk is better
                        if score > grouped[source]["best_score"]:
                            grouped[source]["best_score"] = score

                # Build a list of resumes sorted by best_score
                entries = sorted(grouped.items(), key=lambda x: x[1]["best_score"], reverse=True)[:10]

                # Build DataFrame
                data = []
                for source, info in entries:
                    skills = ", ".join(info["chunks"][0][0].metadata.get('skills', []))
                    tools = ", ".join(info["chunks"][0][0].metadata.get('tools', []))
                    exp_years = info["chunks"][0][0].metadata.get('experience_years', None)
                    data.append({
                        "Source": source,
                        "Score": round(info["best_score"], 4),
                        "Skills": skills,
                        "Tools": tools,
                        "Experience": exp_years
                    })
                df = pd.DataFrame(data)
                st.dataframe(df)

                # Inline expanders: show all chunks per resume
                for source, info in entries:
                    with st.expander(f"{source} (Best Score: {info['best_score']:.4f})"):
                        for doc, score in info["chunks"]:
                            st.write(f"• Score: {score:.4f}")
                            st.write(doc.page_content)
                            st.write("---")

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
            st.markdown("**Generated Job Description & Roles:**")
            st.write(output)

            # Button to reset for a new resume
            if st.button("New Resume"):
                st.session_state["candidate_resume"] = None
                st.experimental_rerun()

# Admin Panel: ingestion & reset
with tabs[2]:
    st.subheader("Upload & Ingest Documents")
    # Ingestion UI (reuse existing sidebar ingestion code here)
    admin_uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="admin_docs_upload"
    )
    # Let user pick the actual Pinecone index names from the .env
    index_choice = st.selectbox(
        "Select index to ingest into",
        (INDEX_RESUME, INDEX_JD),
        label_visibility="visible"
    )
    st.write(f"Ingesting into Pinecone index: **{index_choice}**")
    if st.button("Ingest to Index"):
        if not admin_uploaded_files:
            st.warning("Please upload at least one file.")
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
                        st.error(f"Failed to ingest vectors: {e}. This usually means your index dimension ({INDEX_RESUME} or {INDEX_JD}) doesn't match 1536. Please create a new Pinecone index with dimension=1536 and update your .env accordingly.")
                        continue  # Skip this file and continue with the next one

                    # Clean up the temporary file
                    os.remove(temp_path)
                # Display ingestion summary
                st.info(f"Ingested a total of {total_chunks} document chunks into the '{index_choice}' index.")
                # Show index stats
                stats = (resume_index if index_choice == INDEX_RESUME else jd_index).describe_index_stats()
                st.write("Current index stats:", stats)
            st.success("Ingestion complete!")

    st.markdown("---")
    st.subheader("Reset Index Contents")
    reset_target = st.selectbox("Select index to reset", ("Resume Index", "JD Index"))
    if st.button("Reset Vectors"):
        with st.spinner("Resetting vectors..."):
            if reset_target == "Resume Index":
                resume_index.delete(delete_all=True)
            else:
                jd_index.delete(delete_all=True)
        st.success(f"{reset_target} contents have been deleted.")

# Sidebar note
st.sidebar.markdown("---")
st.sidebar.info("Ai-Cruiter now uses the Pinecone client & OpenAI API.\nEnsure your .env is correct before running.")
