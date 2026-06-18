import os
import shutil
import json
from utils import extract_text_from_pdf, extract_text_from_txt, generate_pdf_report
from rag import index_documents, retrieve_context

# Mock data for testing PDF generation and RAG
MOCK_ANALYSIS = {
    "match_scores": {
        "overall": 82,
        "skills": 85,
        "experience": 75,
        "education": 90
    },
    "missing_skills": {
        "required": ["Java", "Spring Boot", "Docker", "AWS", "SQL", "Git"],
        "found": ["Java", "SQL", "Git"],
        "missing": ["Spring Boot", "Docker", "AWS"]
    },
    "interview_questions": [
        {"question": "Explain Docker architecture and containers.", "answer": "Docker uses a client-server architecture. The Docker client talks to the Docker daemon, which does the heavy lifting of building, running, and distributing containers..."},
        {"question": "What is dependency injection in Spring Boot?", "answer": "Dependency Injection (DI) is a design pattern in which an object receives other objects that it depends on. In Spring, the IoC Container manages DI..."}
    ],
    "learning_roadmap": [
        {"week": "Week 1", "topic": "Spring Boot Basics", "resources_and_actions": ["Study Spring Core concepts", "Build a simple Hello World REST controller"]},
        {"week": "Week 2", "topic": "Docker", "resources_and_actions": ["Understand containers vs VMs", "Write a Dockerfile for a Spring Boot application"]}
    ],
    "ats_score_prediction": {
        "score": 85,
        "feedback": [
            "Use standard section headers like 'Work Experience' and 'Education'.",
            "Incorporate more verbs like 'Led', 'Developed', and 'Architected'."
        ]
    },
    "resume_improvements": [
        {
            "section": "Experience",
            "current_text": "Worked on a web application using Java.",
            "suggested_text": "Developed a high-performance RESTful API using Java and Spring Boot, reducing response time by 15%.",
            "rationale": "Uses action verbs and provides quantifiable business impact."
        }
    ],
    "keyword_highlighting": {
        "matching_keywords": ["Java", "SQL", "Git"],
        "missing_keywords": ["Docker", "AWS", "Spring Boot"]
    }
}

def create_sample_files():
    """Creates a temporary sample text job description."""
    print("Creating sample job description TXT file...")
    jd_content = """
    Job Title: Backend Engineer
    Required Skills: Java, Spring Boot, Docker, AWS, SQL, Git
    Experience: 2+ years of experience in backend development.
    Education: BS in Computer Science or equivalent.
    """
    os.makedirs("jobs", exist_ok=True)
    jd_path = "jobs/sample_jd.txt"
    with open(jd_path, "w", encoding="utf-8") as f:
        f.write(jd_content)
    return jd_path

def test_pipeline():
    print("=== Starting Verification Tests ===")
    
    # 1. Test Text Extraction
    jd_path = create_sample_files()
    jd_text = extract_text_from_txt(jd_path)
    print(f"✓ Text Extraction Test Passed! Read {len(jd_text)} chars from sample JD.")
    
    # Sample Resume text for test
    resume_text = """
    John Doe - Software Engineer
    Skills: Java, SQL, Git, HTML, CSS.
    Experience: Developed web portals using Java. Managed SQL databases.
    Education: Bachelor of Science in Computer Science, 2024.
    """
    
    # 2. Test RAG Indexing & Retrieval
    print("\nIndexing documents in ChromaDB...")
    try:
        # Index with Mock Embeddings (HuggingFace)
        db = index_documents(resume_text, jd_text, api_key=None, use_gemini_embeddings=False)
        print("✓ ChromaDB indexing successful.")
        
        # Query context
        context = retrieve_context(db)
        print(f"✓ RAG context retrieval successful. Retrieved {len(context)} chars.")
        print("Sample context retrieved:")
        print("-" * 40)
        print(context[:300] + "...")
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ RAG Pipeline Test Failed: {e}")
        return False
        
    # 3. Test PDF Report Generation
    print("\nGenerating PDF Report...")
    pdf_path = "test_analysis_report.pdf"
    try:
        generate_pdf_report(MOCK_ANALYSIS, "sample_resume.pdf", "sample_jd.txt", pdf_path)
        if os.path.exists(pdf_path):
            print(f"✓ PDF report successfully generated: {pdf_path} (Size: {os.path.getsize(pdf_path)} bytes)")
            # Clean up
            os.remove(pdf_path)
        else:
            print("❌ PDF Report generation failed: File was not created.")
            return False
    except Exception as e:
        print(f"❌ PDF Report generation failed with exception: {e}")
        return False

    # Clean up jobs folder
    if os.path.exists("jobs"):
        shutil.rmtree("jobs")
        
    print("\n=== All Local Unit Tests Passed Successfully! ===")
    return True

if __name__ == "__main__":
    test_pipeline()
