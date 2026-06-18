import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Create target directories if they don't exist
os.makedirs("resumes", exist_ok=True)
os.makedirs("jobs", exist_ok=True)

def generate_sample_resume(output_path):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#0F172A'), # Slate 900
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#475569'), # Slate 600
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )

    story = []
    
    # Header
    story.append(Paragraph("Alex Mercer", title_style))
    story.append(Paragraph("San Francisco, CA | alex.mercer@email.com | (555) 123-4567 | github.com/alexmercer | linkedin.com/in/alexmercer", subtitle_style))
    story.append(Spacer(1, 5))
    
    # Summary
    story.append(Paragraph("Professional Summary", h1_style))
    story.append(Paragraph("Enthusiastic Software Engineer with 2 years of experience building scalable web applications. Strong backend foundations in Java, SQL, Git, and RESTful API development. Passionate about learning cloud native patterns, containers, and modern architectures.", body_style))
    
    # Skills
    story.append(Paragraph("Technical Skills", h1_style))
    story.append(Paragraph("<b>Languages:</b> Java, SQL, JavaScript, HTML, CSS<br/><b>Frameworks & Libs:</b> Hibernate, JUnit, ExpressJS<br/><b>Developer Tools:</b> Git, GitHub, Maven, Gradle, IntelliJ Idea, VS Code", body_style))
    
    # Work Experience
    story.append(Paragraph("Professional Experience", h1_style))
    story.append(Paragraph("<b>Associate Software Engineer</b> | WebTech Solutions (June 2024 - Present)", body_style))
    story.append(Paragraph("• Developed backend API endpoints using Java, maintaining high code quality and test coverage.<br/>• Optimized relational database queries, improving performance on dashboard operations by 20%.<br/>• Managed code changes and collaborative workflows using Git and GitHub pull requests.", body_style))
    
    story.append(Spacer(1, 5))
    story.append(Paragraph("<b>Software Engineer Intern</b> | DevLaunch Studio (Sept 2023 - May 2024)", body_style))
    story.append(Paragraph("• Assisted in developing web applications and database integrations using Node.js and SQL.<br/>• Collaborated with senior engineers to implement UI components using vanilla HTML/CSS and JavaScript.", body_style))
    
    # Education
    story.append(Paragraph("Education", h1_style))
    story.append(Paragraph("<b>Bachelor of Science in Computer Science</b> | State University (2020 - 2024)", body_style))
    story.append(Paragraph("• GPA: 3.8/4.0<br/>• Key Coursework: Software Design, Database Systems, Web Development, Algorithms & Data Structures", body_style))

    doc.build(story)
    print(f"Generated PDF Resume at: {output_path}")

# Generate Job Descriptions
def create_job_descriptions():
    # JD 1: Backend Engineer (A perfect RAG candidate - requires Docker, AWS, Spring Boot which the resume lacks)
    backend_jd = """Job Title: Backend Software Engineer (Spring Boot & AWS)
Location: Remote / Hybrid (San Francisco, CA)
Job Type: Full-time

About the Role:
We are looking for a Backend Engineer to join our core services team. You will design, build, and deploy cloud-native microservices that power our platform.

Key Responsibilities:
- Design and implement microservices in Java using Spring Boot.
- Deploy and manage applications in containerized environments using Docker.
- Leverage AWS services (EC2, S3, RDS, ECS) to build scalable and resilient infrastructure.
- Write unit, integration, and performance tests to ensure application reliability.
- Manage source control pipelines and CI/CD workflows.

Required Qualifications & Skills:
- BS/MS in Computer Science or related fields.
- Strong proficiency in Java and modern object-oriented programming.
- Experience with Spring Boot framework is highly desired.
- Hands-on experience with containerization tools (Docker).
- Familiarity with cloud platforms, preferably Amazon Web Services (AWS).
- Strong understanding of SQL databases and version control (Git).
"""
    with open("jobs/backend_engineer_jd.txt", "w", encoding="utf-8") as f:
        f.write(backend_jd.strip())
    print("Generated Backend JD at: jobs/backend_engineer_jd.txt")

    # JD 2: Frontend Engineer (React/TypeScript - a low match for our Java resume)
    frontend_jd = """Job Title: Frontend Engineer (React & TypeScript)
Location: San Francisco, CA
Job Type: Full-time

About the Role:
We are seeking a creative Frontend Engineer to build interactive, pixel-perfect user interfaces and dashboards for our web applications.

Required Qualifications & Skills:
- 2+ years of frontend development experience.
- Deep expertise in JavaScript, HTML5, and CSS3.
- Hands-on commercial experience with React.js, Next.js, and TypeScript.
- Experience styling with Tailwind CSS or CSS Modules.
- Experience using state management libraries (Redux Toolkit, Zustand).
- Version control using Git.
"""
    with open("jobs/frontend_engineer_jd.txt", "w", encoding="utf-8") as f:
        f.write(frontend_jd.strip())
    print("Generated Frontend JD at: jobs/frontend_engineer_jd.txt")

    # JD 3: Full Stack Developer (Medium match - requires React & Spring Boot)
    fullstack_jd = """Job Title: Full Stack Developer (Spring Boot + React)
Location: San Francisco, CA

Requirements:
- Strong skills in Java, Spring Boot, SQL, and database design.
- Practical experience with React.js and frontend styling.
- Experience using Git for collaboration.
- Experience deploying application stacks via Docker containers on AWS.
"""
    with open("jobs/fullstack_developer_jd.txt", "w", encoding="utf-8") as f:
        f.write(fullstack_jd.strip())
    print("Generated Full Stack JD at: jobs/fullstack_developer_jd.txt")

if __name__ == "__main__":
    generate_sample_resume("resumes/sample_software_engineer_resume.pdf")
    create_job_descriptions()
    print("--- Test Data Initialization Complete! ---")
