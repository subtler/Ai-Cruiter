# Ai-Cruiter

A smart, AI-powered resume matcher that extracts skills from job descriptions and ranks candidate resumes based on a combination of semantic similarity and skill overlap. Built with Streamlit for an interactive UI and Docker-ready for easy deployment.

---

## 🚀 Table of Contents

1. [Features](#features)  
2. [How It Works](#how-it-works)  
3. [Technology Stack](#technology-stack)  
4. [Getting Started](#getting-started)  
   - [Installation](#installation)  
   - [Configuration](#configuration)  
   - [Index Initialization](#index-initialization)  
   - [Running Locally](#running-locally)  
   - [Running with Docker](#running-with-docker)  
5. [Implementation (Usage)](#implementation-usage)  
6. [CLI Utilities](#cli-utilities)  
7. [Project Structure](#project-structure)  
8. [Contributing](#contributing)  
9. [License](#license)  

---

## 🔥 Features

- **NER-based skill extraction** from Job Descriptions using a pre-trained BERT model  
- **Semantic similarity** matching with vector embeddings (HuggingFace + Pinecone)  
- **Skill overlap scoring** to boost candidates who share explicit skills  
- **Interactive Streamlit UI** for uploading JDs and resumes, viewing top matches, and downloading results  
- **Conversational mode**: chat with memory over indexed resumes (optional)  
- **Auto-create Pinecone indexes** if they don’t exist  
- **CLI scripts** for batch indexing and terminal-based semantic search  
- **Dockerized** for one-step deployment  

---

## 🧠 How It Works

1. **Upload a Job Description**: The app extracts key skills/entities using a BERT-based NER pipeline.  
2. **Upload Resumes**: Each resume is split into chunks, metadata-tagged, and embedded via HuggingFace’s `all-MiniLM-L6-v2`.  
3. **Indexing**: Vectors are stored in Pinecone under `resume-index`. If the index is missing, the app auto-creates it.  
4. **Matching**:  
   - Compute cosine similarity between the JD embedding and each resume chunk.  
   - Compute a skill-overlap score (# shared skills / # JD skills).  
   - Final score = 0.7 * cosine + 0.3 * overlap.  
   - Deduplicate by candidate, rank, and display the top 3 matches first.  
5. **Download Results**: Export a CSV of all ranked candidates with scores and skills.  

---

## 🛠 Technology Stack

- **Language & Frameworks**:  
  - Python 3.10  
  - Streamlit (UI)  
  - Docker (containerization)

- **NLP & Embeddings**:  
  - HuggingFace Transformers (`dslim/bert-base-NER` for NER)  
  - HuggingFace `all-MiniLM-L6-v2` (vector embeddings)  
  - LangChain (pipeline orchestration)

- **Vector Database**:  
  - Pinecone (serverless, cosine similarity)

- **Utilities & CLI**:  
  - `python-dotenv` (environment variables)  
  - `scikit-learn` (cosine_similarity)  
  - Jupyter notebooks for prototype experiments  

---

## ⚙️ Getting Started

Installation & Setup

Follow these steps to get Ai-Cruiter up and running on your local machine or in Docker.

⸻

1. Clone the Repository

git clone https://github.com/<your-username>/Ai-Cruiter.git
cd Ai-Cruiter

2. Create & Activate a Python Virtual Environment

python3 -m venv .venv
source .venv/bin/activate     # on macOS/Linux
.venv\Scripts\activate        # on Windows

3. Install Dependencies

pip install --upgrade pip
pip install -r requirements.txt

4. Configure Environment Variables
    1.    Copy the example file:

cp .env.example .env


    2.    Open .env in your editor and set your Pinecone API key:

PINECONE_API_KEY=your_pinecone_api_key_here



5. (Optional) Manual Index Initialization

Ai-Cruiter will auto-create the required Pinecone indexes (resume-index and jd-index) on first run. To initialize manually instead:

python create_index.py

6. Running the App Locally

streamlit run app.py

    •    Visit http://localhost:8501 in your browser.

7. Running with Docker
    1.    Build the Docker image:

docker build -t resume-matcher .


    2.    Run the container (loads .env automatically):

docker run --env-file .env -p 8501:8501 resume-matcher


    3.    Open http://localhost:8501 in your browser.

⸻

Your installation is complete! You can now upload JDs and resumes to see AI-powered matching in action.
