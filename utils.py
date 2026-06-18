import os
import json
import sqlite3
from datetime import datetime
from pypdf import PdfReader
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ----------------------------------------------------
# 1. Text Extraction
# ----------------------------------------------------
def extract_text_from_pdf(pdf_file) -> str:
    """Extracts text from a PDF file (path or file-like object)."""
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF: {str(e)}")

def extract_text_from_txt(txt_file) -> str:
    """Extracts text from a TXT file (path or file-like object)."""
    try:
        if isinstance(txt_file, str):
            with open(txt_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        else:
            # File-like object (e.g. Streamlit UploadedFile)
            return txt_file.read().decode("utf-8").strip()
    except Exception as e:
        raise ValueError(f"Error reading TXT file: {str(e)}")

# ----------------------------------------------------
# 2. SQLite History Management
# ----------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "analysis_history.db")

def init_db():
    """Initializes the database schema if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            resume_name TEXT,
            jd_name TEXT,
            match_score INTEGER,
            analysis_result TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_to_history(resume_name: str, jd_name: str, match_score: int, analysis_result: dict):
    """Saves a new analysis result to SQLite."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO history (timestamp, resume_name, jd_name, match_score, analysis_result) VALUES (?, ?, ?, ?, ?)",
        (timestamp, resume_name, jd_name, match_score, json.dumps(analysis_result))
    )
    conn.commit()
    conn.close()

def get_history():
    """Retrieves all previous analyses from SQLite, ordered by timestamp desc."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, resume_name, jd_name, match_score, analysis_result FROM history ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    
    history_list = []
    for row in rows:
        history_list.append({
            "id": row[0],
            "timestamp": row[1],
            "resume_name": row[2],
            "jd_name": row[3],
            "match_score": row[4],
            "analysis_result": json.loads(row[5])
        })
    return history_list

def delete_history_item(item_id: int):
    """Deletes an item from the history table."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM history WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()

# ----------------------------------------------------
# 3. Gemini API Integration
# ----------------------------------------------------
def query_gemini_analysis(resume_text: str, jd_text: str, retrieved_context: str, api_key: str, model_name: str = "gemini-1.5-flash") -> dict:
    """Sends prompt containing the resume, job description, and retrieved context to Gemini and expects structured JSON output."""
    genai.configure(api_key=api_key)
    
    prompt = f"""
    You are an expert technical recruiter and career coach. Analyze the candidate's resume against the Job Description.
    Use the provided retrieved chunks of relevant information from both documents to ground your analysis.

    [Retrieved Relevant Context]
    {retrieved_context}

    [Full Resume]
    {resume_text}

    [Full Job Description]
    {jd_text}

    Your response must be a valid JSON object. Do not include markdown formatting like ```json or ```. Return strictly the raw JSON object containing the following keys:
    1. "match_scores": {{ "overall": int, "skills": int, "experience": int, "education": int }}
    2. "missing_skills": {{ "required": [string], "found": [string], "missing": [string] }}
    3. "interview_questions": [ {{ "question": string, "answer": string }} ] (Provide 3-5 realistic interview questions targetting their gaps and core JD requirements, along with brief sample answers)
    4. "learning_roadmap": [ {{ "week": string, "topic": string, "resources_and_actions": [string] }} ] (A structured 4-week roadmap to bridge the missing skills)
    5. "ats_score_prediction": {{ "score": int, "feedback": [string] }} (ATS-compatibility score from 0 to 100, and actionable formatting/keyword feedback)
    6. "resume_improvements": [ {{ "section": string, "current_text": string, "suggested_text": string, "rationale": string }} ] (Specific suggestions for rewriting resume points to align with the JD)
    7. "keyword_highlighting": {{ "matching_keywords": [string], "missing_keywords": [string] }} (Keywords present in both, and keywords present in JD but missing in Resume)

    Double-check that the JSON is fully parseable. Do not add any text before or after the JSON.
    """
    
    model = genai.GenerativeModel(model_name)
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Clean up any accidental markdown formatting if the model still includes it
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        return json.loads(text)
    except json.JSONDecodeError as jde:
        # Fallback parsing in case the LLM returned extra text
        try:
            start_idx = text.find("{")
            end_idx = text.rfind("}") + 1
            if start_idx != -1 and end_idx != -1:
                return json.loads(text[start_idx:end_idx])
            else:
                raise jde
        except Exception as inner_e:
            raise ValueError(f"Failed to parse Gemini response as JSON: {text}. Error: {str(inner_e)}")
    except Exception as e:
        raise ValueError(f"Gemini API Error: {str(e)}")

# ----------------------------------------------------
# 4. ReportLab PDF Generation
# ----------------------------------------------------
def generate_pdf_report(analysis: dict, resume_name: str, jd_name: str, output_path: str):
    """Generates a professional, premium-looking PDF report of the analysis."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles for Premium Feel
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#0F172A'), # Slate 900
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#64748B'), # Slate 500
        spaceAfter=25
    )
    
    h1_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#0284C7'), # Sky 600
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'SubSectionHeading',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#334155'), # Slate 700
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'), # Slate 700
        spaceAfter=8
    )
    
    bold_body_style = ParagraphStyle(
        'BoldBodyTextCustom',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    table_text_style = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#334155')
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.white
    )

    story = []
    
    # 1. Header Section
    story.append(Paragraph("AI Resume Analysis Report", title_style))
    story.append(Paragraph(f"<b>Resume:</b> {resume_name} | <b>Job Description:</b> {jd_name}<br/><b>Generated on:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", subtitle_style))
    story.append(Spacer(1, 10))
    
    # 2. Scores Overview (Table layout)
    story.append(Paragraph("Match & ATS Scores", h1_style))
    scores = analysis.get("match_scores", {})
    ats = analysis.get("ats_score_prediction", {})
    
    score_data = [
        [Paragraph("Metric", table_header_style), Paragraph("Score", table_header_style), Paragraph("Assessment", table_header_style)],
        [Paragraph("Overall Match", table_text_style), Paragraph(f"{scores.get('overall', 0)}%", bold_body_style), Paragraph("General alignment with job requirements", table_text_style)],
        [Paragraph("Skills Match", table_text_style), Paragraph(f"{scores.get('skills', 0)}%", table_text_style), Paragraph("Technical and soft skills similarity", table_text_style)],
        [Paragraph("Experience Match", table_text_style), Paragraph(f"{scores.get('experience', 0)}%", table_text_style), Paragraph("Years and quality of relevant experience", table_text_style)],
        [Paragraph("Education Match", table_text_style), Paragraph(f"{scores.get('education', 0)}%", table_text_style), Paragraph("Degree and domain alignment", table_text_style)],
        [Paragraph("ATS Score Prediction", table_text_style), Paragraph(f"{ats.get('score', 0)}%", bold_body_style), Paragraph("ATS parsing compatibility and format check", table_text_style)],
    ]
    
    score_table = Table(score_data, colWidths=[150, 80, 290])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')), # Slate 800
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')), # Slate 300
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8FAFC')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 15))
    
    # 3. Skills Analysis
    story.append(Paragraph("Skills Analysis", h1_style))
    skills = analysis.get("missing_skills", {})
    story.append(Paragraph("<b>Required Skills:</b> " + ", ".join(skills.get("required", [])), body_style))
    story.append(Paragraph("<b>Found Skills:</b> " + ", ".join(skills.get("found", [])), body_style))
    story.append(Paragraph("<b>Missing Skills:</b> " + ", ".join(skills.get("missing", [])), body_style))
    story.append(Spacer(1, 15))
    
    # 4. ATS Feedback & Suggestions
    story.append(Paragraph("ATS Feedback & Enhancements", h1_style))
    for fb in ats.get("feedback", []):
        story.append(Paragraph(f"• {fb}", body_style))
    story.append(Spacer(1, 15))
    
    # 5. Resume Improvements
    improvements = analysis.get("resume_improvements", [])
    if improvements:
        story.append(Paragraph("Specific Resume Line Improvements", h1_style))
        imp_data = [[
            Paragraph("Section", table_header_style), 
            Paragraph("Current Text", table_header_style), 
            Paragraph("Suggested Text & Rationale", table_header_style)
        ]]
        for imp in improvements:
            imp_data.append([
                Paragraph(imp.get('section', 'Experience'), table_text_style),
                Paragraph(imp.get('current_text', ''), table_text_style),
                Paragraph(f"<b>Suggested:</b> {imp.get('suggested_text', '')}<br/><b>Why:</b> {imp.get('rationale', '')}", table_text_style)
            ])
        imp_table = Table(imp_data, colWidths=[80, 200, 240])
        imp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8FAFC')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(imp_table)
        story.append(Spacer(1, 15))
        
    story.append(PageBreak()) # Clean page break for interview questions and roadmap
    
    # 6. Interview Preparation
    story.append(Paragraph("Generated Interview Questions", h1_style))
    questions = analysis.get("interview_questions", [])
    for idx, q in enumerate(questions, 1):
        story.append(Paragraph(f"<b>Q{idx}. {q.get('question', '')}</b>", h2_style))
        story.append(Paragraph(f"<b>Suggested Answer:</b> {q.get('answer', '')}", body_style))
        story.append(Spacer(1, 10))
        
    story.append(Spacer(1, 15))
    
    # 7. Learning Roadmap
    story.append(Paragraph("4-Week Learning Roadmap", h1_style))
    roadmap = analysis.get("learning_roadmap", [])
    for rm in roadmap:
        story.append(Paragraph(f"<b>{rm.get('week', '')}: {rm.get('topic', '')}</b>", h2_style))
        for act in rm.get("resources_and_actions", []):
            story.append(Paragraph(f"• {act}", body_style))
        story.append(Spacer(1, 8))
        
    # Build Document
    doc.build(story)
