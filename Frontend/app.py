import streamlit as st
from datetime import datetime
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)



st.set_page_config(
    page_title="AI Resume Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { display:none; }
    header { visibility:hidden; }
    html { scroll-behavior:smooth; }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --dark:#2f4f3a;
  --mid:#3f6b4f;
  --light:#6a9c78;
  --cream:#fefae0;
  --text:#1f2933;
  --muted:#6b7280;
}

* {
  font-family:Inter,sans-serif;
  box-sizing:border-box;
}

body {
  background:
    radial-gradient(circle at 15% 10%, rgba(106,156,120,.08), transparent 40%),
    radial-gradient(circle at 85% 0%, rgba(63,107,79,.06), transparent 35%),
    linear-gradient(to bottom,#f8faf8,#fefae0);
  color:var(--text);
}

/* ---------- NAVBAR ---------- */
.navbar {
  position:fixed;
  inset:0 0 auto 0;
  padding:12px 56px;
  background:rgba(255,255,255,.9);
  backdrop-filter:blur(14px);
  display:flex;
  justify-content:space-between;
  align-items:center;
  z-index:1000;
  box-shadow:0 4px 18px rgba(0,0,0,.06);
}

.navbar strong {
  font-weight:700;
  letter-spacing:-.02em;
}

.navbar a {
  text-decoration:none;
  color:var(--mid);
  font-weight:500;
  margin-left:22px;
  transition:opacity .2s ease;
}

.navbar a:hover {
  opacity:.75;
}

.nav-btn {
  background:var(--mid);
  color:white !important;
  padding:8px 18px;
  border-radius:999px;
  font-weight:600;
}

/* ---------- HERO ---------- */
.hero {
  margin-top:110px;
  background:linear-gradient(135deg,var(--dark),var(--mid));
  padding:84px 72px;
  border-radius:32px;
  color:white;
  box-shadow:0 28px 60px rgba(0,0,0,.22);
}

.hero h1 {
  font-size:44px;
  line-height:1.15;
  margin-bottom:14px;
  letter-spacing:-.02em;
}

.hero p {
  font-size:18px;
  max-width:720px;
  opacity:.95;
}

.btn {
  display:inline-block;
  background:var(--cream);
  color:var(--mid);
  padding:13px 26px;
  border-radius:14px;
  font-weight:600;
  margin-right:12px;
  text-decoration:none;
  transition:transform .15s ease, box-shadow .15s ease;
}

.btn:hover {
  transform:translateY(-1px);
  box-shadow:0 6px 18px rgba(0,0,0,.15);
}

/* ---------- SECTIONS ---------- */
.section {
  max-width:1120px;
  margin:auto;
  padding:84px 40px;
}

.section h2 {
  font-size:34px;
  margin-bottom:28px;
  color:var(--mid);
  letter-spacing:-.02em;
}

/* ---------- CARDS ---------- */
.grid {
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
  gap:26px;
}

.card {
  background:white;
  padding:28px;
  border-radius:20px;
  box-shadow:0 14px 34px rgba(0,0,0,.08);
  transition:transform .15s ease, box-shadow .15s ease;
}

.card:hover {
  transform:translateY(-2px);
  box-shadow:0 18px 44px rgba(0,0,0,.12);
}

.card h3 {
  font-size:19px;
  margin-bottom:8px;
  color:var(--mid);
}

.card p {
  font-size:15px;
  line-height:1.6;
  color:var(--muted);
}

/* ---------- CTA ---------- */
.cta {
  background:linear-gradient(135deg,var(--dark),var(--light));
  padding:86px 70px;
  border-radius:34px;
  color:white;
  text-align:center;
  box-shadow:0 28px 70px rgba(0,0,0,.28);
}

.cta h2 {
  font-size:38px;
  letter-spacing:-.02em;
}

/* ---------- FOOTER ---------- */
footer {
  text-align:center;
  padding:56px;
  color:#6b7280;
  border-top:1px solid rgba(0,0,0,.08);
  font-size:14px;
}
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="navbar">
  <strong>AI Resume Intelligence</strong>
  <div>
    <a href="#how">How It Works</a>
    <a href="#different">Why Different</a>
    <a class="nav-btn" href="?page=analyzer">Analyze Resume</a>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="hero">
  <h1>See Your Resume Through a Recruiter’s Lens</h1>
  <p>
    Semantic ML analysis of skills, experience depth,
    and real hiring signals — not keyword stuffing.
  </p><br/>
  <a class="btn" href="?page=analyzer">Analyze My Resume</a>
  <a class="btn" href="#how">How It Works</a>
</div>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="section">
  <h2>Who Is This For?</h2>
  <div class="grid">
    <div class="card">
      <h3>Job Seekers</h3>
      <p>Identify real gaps before applying and improve faster.</p>
    </div>
    <div class="card">
      <h3>Recruiters</h3>
      <p>Objective, explainable signals beyond ATS keyword matching.</p>
    </div>
    <div class="card">
      <h3>Career Coaches</h3>
      <p>Data-backed resume feedback instead of intuition.</p>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div id="how" class="section">
  <h2>How It Works</h2>
  <div class="grid">
    <div class="card">
      <h3>Resume Ingestion</h3>
      <p>PDF and text resumes are parsed into structured sections.</p>
    </div>
    <div class="card">
      <h3>Semantic ML Engine</h3>
      <p>Sentence-transformer embeddings capture context and depth.</p>
    </div>
    <div class="card">
      <h3>Explainable Insights</h3>
      <p>Scores, ablation analysis, and improvement roadmap.</p>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div id="different" class="section">
  <h2>Why This Is Different</h2>
  <div class="grid">
    <div class="card"><h3>Semantic Understanding</h3><p>Understands how skills are used, not just mentioned.</p></div>
    <div class="card"><h3>Section-Aware Scoring</h3><p>Projects and experience matter more than lists.</p></div>
    <div class="card"><h3>Explainable Gaps</h3><p>Clear reasons for missing or weak skills.</p></div>
    <div class="card"><h3>Ablation Analysis</h3><p>Measures real contribution of resume sections.</p></div>
    <div class="card"><h3>Skill Roadmap</h3><p>Personalized next steps, not generic advice.</p></div>
    <div class="card"><h3>Recruiter-Oriented</h3><p>Built to mirror hiring workflows and ATS logic.</p></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="cta">
  <h2>Stop Guessing. Start Improving.</h2>
  <p>Understand exactly how hiring systems interpret your resume.</p><br/>
  <a class="btn" href="?page=analyzer">Analyze My Resume →</a>
  <p style="font-size:14px;opacity:.85;margin-top:16px;">
    No login • No data stored • Local processing
  </p>
</div>
""",
    unsafe_allow_html=True,
)

year = datetime.now().year
st.markdown(
    f"""
<footer>
  <strong>AI Resume Intelligence</strong><br/>
  © {year}
</footer>
""",
    unsafe_allow_html=True,
)

page = st.query_params.get("page")
if page == "analyzer":
    st.switch_page("pages/1_Analyzer.py")
