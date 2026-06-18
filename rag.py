import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# Chroma database persistent folder
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

def get_embeddings(api_key=None, use_gemini=False):
    """Initializes and returns the embeddings object.
    Can use Google Generative AI Embeddings (API-based, very fast) or
    HuggingFace sentence-transformers (local, standard).
    """
    if use_gemini and api_key:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
        except Exception as e:
            print(f"Failed to load Google Generative AI Embeddings: {e}. Falling back to HuggingFace.")
            
    # Default to HuggingFace
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def clear_vector_db():
    """Removes the persistent Chroma directory to prevent mixing old session data."""
    if os.path.exists(PERSIST_DIR):
        try:
            # Simple clean up of database folder
            shutil.rmtree(PERSIST_DIR)
        except Exception as e:
            print(f"Warning: Could not clear chroma_db folder: {e}")

def index_documents(resume_text: str, jd_text: str, api_key: str = None, use_gemini_embeddings: bool = False):
    """Splits resume and Job Description text, generates embeddings, and indexes them in ChromaDB."""
    # 1. Clear database to avoid bleed-over from previous analyses
    clear_vector_db()
    
    # 2. Text splitter setup
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120
    )
    
    # 3. Create Documents with metadata
    docs = []
    
    # Chunk Resume
    resume_chunks = text_splitter.split_text(resume_text)
    for idx, chunk in enumerate(resume_chunks):
        docs.append(Document(
            page_content=chunk,
            metadata={"source": "resume", "chunk_id": idx}
        ))
        
    # Chunk Job Description
    jd_chunks = text_splitter.split_text(jd_text)
    for idx, chunk in enumerate(jd_chunks):
        docs.append(Document(
            page_content=chunk,
            metadata={"source": "job_description", "chunk_id": idx}
        ))
        
    # 4. Initialize Embeddings
    embeddings = get_embeddings(api_key, use_gemini_embeddings)
    
    # 5. Store in Chroma
    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=PERSIST_DIR
    )
    
    return db

def retrieve_context(db, queries=None) -> str:
    """Retrieves relevant chunks from ChromaDB for a list of target queries and returns a combined context string."""
    if not queries:
        queries = [
            "technical skills, programming languages, libraries, databases, frameworks, developer tools",
            "professional work experience, projects, responsibilities, achievements",
            "education, college degree, certifications, training, academic background",
            "job description requirements, core qualifications, responsibilities, job duties"
        ]
        
    retrieved_chunks = []
    seen_contents = set()
    
    for query in queries:
        # Retrieve top 3 relevant chunks for each aspect
        docs = db.similarity_search(query, k=3)
        for doc in docs:
            content = doc.page_content.strip()
            source = doc.metadata.get("source", "unknown")
            if content not in seen_contents:
                seen_contents.add(content)
                retrieved_chunks.append(f"[{source.upper()} CHUNK]:\n{content}")
                
    return "\n\n".join(retrieved_chunks)
