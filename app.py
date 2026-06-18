import os
import tempfile
import streamlit as st
import pandas as pd
from utils import (
    extract_text_from_pdf,
    extract_text_from_txt,
    save_to_history,
    get_history,
    delete_history_item,
    query_gemini_analysis,
    generate_pdf_report
)
from rag import index_documents, retrieve_context

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="AI Resume & JD Analyzer (RAG)",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "resume_name" not in st.session_state:
    st.session_state.resume_name = ""
if "jd_name" not in st.session_state:
    st.session_state.jd_name = ""
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Single Analysis"

# Custom CSS for Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Font style overrides */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Premium Title and Headers */
    .title-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    .title-container::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 200%;
        background: radial-gradient(circle, rgba(14,165,233,0.1) 0%, transparent 70%);
        pointer-events: none;
    }
    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-subtitle {
        font-size: 16px;
        color: #94A3B8;
        margin-top: 8px;
        margin-bottom: 0;
        font-weight: 300;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s, box-shadow 0.2s;
        text-align: center;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    .metric-title {
        font-size: 14px;
        color: #64748B;
        font-weight: 500;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        color: #0F172A;
    }
    
    /* Badges & Tags */
    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-success {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #BBF7D0;
    }
    .badge-warning {
        background-color: #FEF9C3;
        color: #A16207;
        border: 1px solid #FEF08A;
    }
    .badge-danger {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
    }
    .badge-info {
        background-color: #E0F2FE;
        color: #0369A1;
        border: 1px solid #BAE6FD;
    }
    
    /* Timeline / Roadmap items */
    .roadmap-card {
        background-color: #F8FAFC;
        border-left: 4px solid #0284C7;
        padding: 16px;
        border-radius: 0 12px 12px 0;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .roadmap-week {
        font-size: 12px;
        font-weight: 700;
        color: #0284C7;
        text-transform: uppercase;
    }
    .roadmap-topic {
        font-size: 16px;
        font-weight: 600;
        color: #0F172A;
        margin: 4px 0 8px 0;
    }
    
    /* Highlight containers */
    .highlight-box {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        line-height: 1.6;
    }
    
    /* Styled lists */
    ul.check-list {
        list-style-type: none;
        padding-left: 0;
    }
    ul.check-list li::before {
        content: "✓  ";
        color: #10B981;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Sidebar Section
# ----------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/resume.png", width=64)
    st.markdown("### Configuration & History")
    
    # API Key Input
    api_key_env = os.environ.get("GEMINI_API_KEY", "")
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        value=api_key_env if api_key_env else "",
        placeholder="Enter your AIzaSy... key",
        help="If empty, the app will try to read from the environment variable GEMINI_API_KEY."
    )
    
    if not api_key:
        st.warning("⚠️ Enter Gemini API Key to perform analysis.")
    else:
        st.success("🔑 API Key configured.")
        
    st.markdown("---")
    
    # RAG Settings
    st.markdown("### RAG Pipeline Settings")
    use_gemini_embeddings = st.checkbox(
        "Use Gemini Embeddings API",
        value=True,
        help="If checked, uses models/embedding-001. If unchecked, uses local sentence-transformers (which downloads ~100MB model locally)."
    )
    
    st.markdown("---")
    
    # Placement Interview helper explanation
    with st.expander("🎓 Placement QA (Why RAG?)", expanded=False):
        st.info("""
        **Q: Why did you use RAG?**
        
        *Answer:* 
        "Instead of sending the entire resume and job description directly to the LLM, I converted them into embeddings and stored them in a vector database (ChromaDB). When the user requests analysis, the system retrieves only the most semantically relevant sections and provides them to the LLM. This reduces token usage, avoids context window bloat, and increases output relevance. This follows the Retrieval-Augmented Generation (RAG) architecture."
        """)
        
    st.markdown("---")
    
    # History section
    st.markdown("### Analysis History")
    history_list = get_history()
    
    if history_list:
        for idx, item in enumerate(history_list):
            col1, col2 = st.columns([4, 1])
            # Select button
            if col1.button(
                f"📄 {item['resume_name'][:12]}.. vs {item['jd_name'][:12]}.. ({item['match_score']}%)",
                key=f"hist_btn_{item['id']}",
                help=f"Loaded at: {item['timestamp']}"
            ):
                st.session_state.analysis_result = item["analysis_result"]
                st.session_state.resume_name = item["resume_name"]
                st.session_state.jd_name = item["jd_name"]
                st.toast(f"Loaded analysis from history!")
            
            # Delete button
            if col2.button("🗑️", key=f"hist_del_{item['id']}", help="Delete history item"):
                delete_history_item(item["id"])
                st.rerun()
    else:
        st.caption("No history found. Complete an analysis to see records here.")

# ----------------------------------------------------
# Main Layout Section
# ----------------------------------------------------
# Title Container
st.markdown("""
<div class="title-container">
    <h1 class="main-title">🚀 AI Resume + Job Description Analyzer</h1>
    <p class="main-subtitle">RAG-powered resume matching, ATS prediction, skills alignment, and personalized interview/learning preparation</p>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_choice = st.tabs(["📊 Single Job Analysis", "⚖️ Multi-Job Comparison"])

# ----------------------------------------------------
# Tab 1: Single Job Analysis
# ----------------------------------------------------
with tab_choice[0]:
    col_upload_left, col_upload_right = st.columns(2)
    
    with col_upload_left:
        st.markdown("### 1. Upload Resume")
        resume_file = st.file_uploader(
            "Choose Resume PDF",
            type=["pdf"],
            help="PDF resume to analyze."
        )
        
    with col_upload_right:
        st.markdown("### 2. Upload Job Description")
        jd_file = st.file_uploader(
            "Choose Job Description (PDF/TXT)",
            type=["pdf", "txt"],
            help="Job posting requirements PDF or plain text."
        )
        
    # Analyze Trigger
    if st.button("🔍 Analyze Resume vs Job Description", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please enter a Google Gemini API Key in the sidebar or environment variable to proceed.")
        elif not resume_file:
            st.error("Please upload a resume (PDF).")
        elif not jd_file:
            st.error("Please upload a Job Description.")
        else:
            with st.spinner("Extracting text and indexing documents into ChromaDB..."):
                try:
                    # Step 1: Text extraction
                    resume_text = extract_text_from_pdf(resume_file)
                    
                    if jd_file.name.endswith(".pdf"):
                        jd_text = extract_text_from_pdf(jd_file)
                    else:
                        jd_text = extract_text_from_txt(jd_file)
                        
                    # Step 2: RAG Pipeline Indexing
                    db = index_documents(
                        resume_text=resume_text,
                        jd_text=jd_text,
                        api_key=api_key,
                        use_gemini_embeddings=use_gemini_embeddings
                    )
                    
                    # Step 3: Retrieve context
                    retrieved_context = retrieve_context(db)
                    
                    st.toast("RAG retrieval complete! Querying Gemini API...")
                    
                    # Step 4: LLM Analysis
                    analysis_result = query_gemini_analysis(
                        resume_text=resume_text,
                        jd_text=jd_text,
                        retrieved_context=retrieved_context,
                        api_key=api_key
                    )
                    
                    # Save local Session State & Database History
                    st.session_state.analysis_result = analysis_result
                    st.session_state.resume_name = resume_file.name
                    st.session_state.jd_name = jd_file.name
                    
                    overall_score = analysis_result.get("match_scores", {}).get("overall", 50)
                    save_to_history(
                        resume_name=resume_file.name,
                        jd_name=jd_file.name,
                        match_score=overall_score,
                        analysis_result=analysis_result
                    )
                    
                    st.success("Analysis complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

    # Display Results if Available
    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        st.markdown("---")
        st.markdown(f"## 📋 Analysis Dashboard: *{st.session_state.resume_name}* vs *{st.session_state.jd_name}*")
        
        # Display match meters
        scores = res.get("match_scores", {})
        ats = res.get("ats_score_prediction", {})
        
        cols_score = st.columns(5)
        metrics = [
            ("Overall Match", scores.get("overall", 0), "#0284C7"),
            ("Skills Match", scores.get("skills", 0), "#10B981"),
            ("Experience Match", scores.get("experience", 0), "#F59E0B"),
            ("Education Match", scores.get("education", 0), "#8B5CF6"),
            ("ATS Score", ats.get("score", 0), "#EC4899")
        ]
        
        for idx, (title, score, color) in enumerate(metrics):
            with cols_score[idx]:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid {color};">
                    <div class="metric-title">{title}</div>
                    <div class="metric-value" style="color: {color};">{score}%</div>
                </div>
                """, unsafe_allow_html=True)
                
        st.markdown("<br/>", unsafe_allow_html=True)
        
        # Detailed dashboard tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Match Scores & ATS Feedback",
            "🕵️ Skills Gap & Keywords",
            "✍️ Resume Line Enhancements",
            "💬 Interview Prep (Q&A)",
            "📅 Learning Roadmap",
            "📥 PDF Report Export"
        ])
        
        # Tab 1: Scores & ATS Feedback
        with tab1:
            col_ats_l, col_ats_r = st.columns([1, 1])
            with col_ats_l:
                st.markdown("### Match Assessment Breakdown")
                st.markdown(f"**Overall Match:** {scores.get('overall', 0)}%")
                st.progress(scores.get("overall", 0) / 100)
                st.markdown(f"**Skills Match:** {scores.get('skills', 0)}%")
                st.progress(scores.get("skills", 0) / 100)
                st.markdown(f"**Experience Match:** {scores.get('experience', 0)}%")
                st.progress(scores.get("experience", 0) / 100)
                st.markdown(f"**Education Match:** {scores.get('education', 0)}%")
                st.progress(scores.get("education", 0) / 100)
                
            with col_ats_r:
                st.markdown("### ATS Score Feedback")
                st.markdown(f"**Predicted ATS Score:** `{ats.get('score', 0)} / 100`")
                for item in ats.get("feedback", []):
                    st.markdown(f"- {item}")
                    
        # Tab 2: Skills Gap
        with tab2:
            st.markdown("### Skills Gaps")
            skills = res.get("missing_skills", {})
            
            col_sk_l, col_sk_m, col_sk_r = st.columns(3)
            
            with col_sk_l:
                st.markdown("#### Found Skills")
                found_skills = skills.get("found", [])
                if found_skills:
                    for s in found_skills:
                        st.markdown(f'<span class="badge badge-success">{s}</span>', unsafe_allow_html=True)
                else:
                    st.caption("No matching skills identified.")
                    
            with col_sk_m:
                st.markdown("#### Missing Skills")
                missing_skills = skills.get("missing", [])
                if missing_skills:
                    for s in missing_skills:
                        st.markdown(f'<span class="badge badge-danger">{s}</span>', unsafe_allow_html=True)
                else:
                    st.caption("No missing skills identified! Great match.")
                    
            with col_sk_r:
                st.markdown("#### Required in Job")
                req_skills = skills.get("required", [])
                if req_skills:
                    for s in req_skills:
                        st.markdown(f'<span class="badge badge-info">{s}</span>', unsafe_allow_html=True)
                else:
                    st.caption("No specific skills extracted from JD.")
                    
            st.markdown("---")
            st.markdown("### Keyword Analysis & Highlighting")
            keyword_data = res.get("keyword_highlighting", {})
            
            col_kw_l, col_kw_r = st.columns(2)
            with col_kw_l:
                st.markdown("#### Matching Keywords")
                match_kws = keyword_data.get("matching_keywords", [])
                for kw in match_kws:
                    st.markdown(f'<span class="badge badge-success">{kw}</span>', unsafe_allow_html=True)
            with col_kw_r:
                st.markdown("#### Missing Keywords (Highly Recommended to Add)")
                miss_kws = keyword_data.get("missing_keywords", [])
                for kw in miss_kws:
                    st.markdown(f'<span class="badge badge-warning">{kw}</span>', unsafe_allow_html=True)
                    
        # Tab 3: Resume Line Enhancements
        with tab3:
            st.markdown("### Specific Resume Formatting and Bullet Point Improvements")
            st.info("💡 Edit the bullet points on your resume to match the suggested rewrites. This improves both ATS parsing and interviewer impressions.")
            
            improvements = res.get("resume_improvements", [])
            if improvements:
                df_improvements = pd.DataFrame(improvements)
                st.table(df_improvements)
            else:
                st.success("No resume rewrites recommended! The current text aligns cleanly with the JD description.")
                
        # Tab 4: Interview Preparation
        with tab4:
            st.markdown("### Generated Interview Questions (Tailored to Gaps)")
            st.caption("Click on any question to view its suggested response.")
            
            questions = res.get("interview_questions", [])
            if questions:
                for idx, q in enumerate(questions, 1):
                    with st.expander(f"Question {idx}: {q.get('question', '')}"):
                        st.markdown(f"**Suggested Response:**\n{q.get('answer', '')}")
            else:
                st.info("No customized interview questions generated.")
                
        # Tab 5: Learning Roadmap
        with tab5:
            st.markdown("### 4-Week Technical Learning Roadmap")
            roadmap = res.get("learning_roadmap", [])
            if roadmap:
                for rm in roadmap:
                    st.markdown(f"""
                    <div class="roadmap-card">
                        <div class="roadmap-week">{rm.get('week', 'Week')}</div>
                        <div class="roadmap-topic">{rm.get('topic', 'Topic')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    for act in rm.get("resources_and_actions", []):
                        st.markdown(f"- {act}")
                    st.markdown("<br/>", unsafe_allow_html=True)
            else:
                st.info("No roadmap generated.")
                
        # Tab 6: Export Report
        with tab6:
            st.markdown("### Download Premium PDF Report")
            st.markdown("You can export this full evaluation dashboard as a professional PDF report containing the match scores, missing skills, resume rewriting tips, interview questions, and the weekly study plan.")
            
            # Temporary file export
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                generate_pdf_report(
                    analysis=res,
                    resume_name=st.session_state.resume_name,
                    jd_name=st.session_state.jd_name,
                    output_path=tmp_file.name
                )
                
                with open(tmp_file.name, "rb") as f:
                    pdf_bytes = f.read()
                    
            st.download_button(
                label="📥 Download Report PDF",
                data=pdf_bytes,
                file_name=f"resume_analysis_report_{st.session_state.resume_name[:-4]}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# ----------------------------------------------------
# Tab 2: Multi-Job Comparison
# ----------------------------------------------------
with tab_choice[1]:
    st.markdown("### Compare Resume against Multiple Job Descriptions")
    st.caption("Upload a single resume and multiple JDs (up to 3) to compare scores side-by-side.")
    
    col_comp_l, col_comp_r = st.columns([1, 2])
    
    with col_comp_l:
        comp_resume = st.file_uploader(
            "Upload Resume PDF",
            type=["pdf"],
            key="comp_res_upload"
        )
        
        comp_jd1 = st.file_uploader(
            "Upload Job Description 1 (PDF/TXT)",
            type=["pdf", "txt"],
            key="comp_jd1_upload"
        )
        
        comp_jd2 = st.file_uploader(
            "Upload Job Description 2 (PDF/TXT)",
            type=["pdf", "txt"],
            key="comp_jd2_upload"
        )
        
        comp_jd3 = st.file_uploader(
            "Upload Job Description 3 (PDF/TXT)",
            type=["pdf", "txt"],
            key="comp_jd3_upload"
        )
        
        run_comparison = st.button("📊 Run Multi-Job Comparison", type="primary", use_container_width=True)
        
    with col_comp_r:
        if run_comparison:
            if not api_key:
                st.error("Please enter a Google Gemini API Key in the sidebar or environment variable to proceed.")
            elif not comp_resume:
                st.error("Please upload your Resume.")
            elif not (comp_jd1 or comp_jd2 or comp_jd3):
                st.error("Please upload at least one Job Description to compare.")
            else:
                jds_to_compare = []
                if comp_jd1: jds_to_compare.append(comp_jd1)
                if comp_jd2: jds_to_compare.append(comp_jd2)
                if comp_jd3: jds_to_compare.append(comp_jd3)
                
                comparison_data = []
                
                with st.spinner(f"Running analysis for {len(jds_to_compare)} job descriptions..."):
                    try:
                        # Extract resume text
                        res_text = extract_text_from_pdf(comp_resume)
                        
                        for jd_file_comp in jds_to_compare:
                            # Extract JD text
                            if jd_file_comp.name.endswith(".pdf"):
                                jd_txt = extract_text_from_pdf(jd_file_comp)
                            else:
                                jd_txt = extract_text_from_txt(jd_file_comp)
                                
                            # RAG Pipeline Indexing for this specific JD
                            db_comp = index_documents(
                                resume_text=res_text,
                                jd_text=jd_txt,
                                api_key=api_key,
                                use_gemini_embeddings=use_gemini_embeddings
                            )
                            
                            # Retrieve context
                            ret_context = retrieve_context(db_comp)
                            
                            # LLM Analysis
                            analysis = query_gemini_analysis(
                                resume_text=res_text,
                                jd_text=jd_txt,
                                retrieved_context=ret_context,
                                api_key=api_key
                            )
                            
                            scores_comp = analysis.get("match_scores", {})
                            ats_comp = analysis.get("ats_score_prediction", {})
                            
                            comparison_data.append({
                                "Job Description": jd_file_comp.name,
                                "Overall Match": scores_comp.get("overall", 0),
                                "Skills Match": scores_comp.get("skills", 0),
                                "Experience Match": scores_comp.get("experience", 0),
                                "Education Match": scores_comp.get("education", 0),
                                "ATS Score": ats_comp.get("score", 0)
                            })
                            
                        # Build DataFrame and render
                        df_comp = pd.DataFrame(comparison_data)
                        
                        st.markdown("### Comparison Results")
                        st.dataframe(df_comp.set_index("Job Description"))
                        
                        # Render Comparison Chart
                        st.markdown("#### Score Comparisons Chart")
                        # Melt dataframe for st.bar_chart or use st.bar_chart directly
                        chart_df = df_comp.set_index("Job Description")
                        st.bar_chart(chart_df)
                        
                        # Find the best fit
                        best_fit = df_comp.loc[df_comp['Overall Match'].idxmax()]
                        st.success(f"🏆 **Best Fit:** **{best_fit['Job Description']}** with an Overall Match of **{best_fit['Overall Match']}%**!")
                    except Exception as e:
                        st.error(f"Comparison failed: {str(e)}")
        else:
            st.info("Upload files and click 'Run Multi-Job Comparison' to visualize side-by-side analysis.")
