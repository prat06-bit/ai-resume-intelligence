import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import umap

from backend.skill_extractor import extract_skills, load_skills
from backend.semantic_matcher import SemanticMatcher
from backend.scoring import compute_score, role_weighted_score
from backend.roadmap import generate_roadmap

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("AI Resume Analyzer")
st.caption("Semantic ML • Sentence Transformers • Explainable ATS")

st.markdown(
    """
    <style>
    .stMetric { text-align: center; }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        padding: 10px;
    }
    .card {
        padding: 16px;
        border-radius: 12px;
        background-color: #f6f6f6;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

role = st.selectbox(
    "Target Role",
    [
        "software_engineer",
        "backend_engineer",
        "frontend_engineer",
        "fullstack_engineer",
        "ml_engineer",
        "data_scientist",
        "data_analyst",
        "devops_engineer",
        "cloud_engineer",
        "student",
        "intern"
    ]
)

col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("Upload Resume (TXT / PDF)", type=["txt", "pdf"])
    resume_text = st.text_area("Or paste resume text", height=220)

with col2:
    jd_text = st.text_area("Paste Job Description", height=350)

if not jd_text:
    st.stop()

def read_resume(file):
    if file.type == "application/pdf":
        import pdfplumber
        with pdfplumber.open(file) as pdf:
            return "\n".join(
                p.extract_text() for p in pdf.pages if p.extract_text()
            )
    return file.read().decode("utf-8")

if resume_file:
    resume_text = read_resume(resume_file)

if not resume_text:
    st.stop()

matcher = SemanticMatcher()
skills_db = load_skills()

resume_skills = extract_skills(resume_text, skills_db)
jd_skills = extract_skills(jd_text, skills_db)

similarity = matcher.match_skills(resume_skills, jd_skills)
if not similarity:
    similarity = {skill: 0.0 for skill in jd_skills}

skill_score, confidence = compute_score(similarity)

section_embeddings = matcher.embed_sections(resume_text)
jd_emb = matcher.embed([jd_text])[0]

section_similarities = {
    sec: float(cosine_similarity([emb], [jd_emb])[0][0])
    for sec, emb in section_embeddings.items()
    if emb is not None
}

final_score = role_weighted_score(section_similarities, role)

matched = {k: v for k, v in similarity.items() if v >= 0.55}
missing = {k: v for k, v in similarity.items() if v < 0.55}
if not matched and not missing:
    missing = similarity.copy()

m1, m2, m3 = st.columns(3)
m1.metric("Final Match Score", f"{final_score:.1f}%")
m2.metric("Matched Skills", len(matched))
m3.metric("Missing Skills", len(missing))
st.caption(f"Confidence ±{confidence:.1f}%")

tab1, tab2, tab3, tab4 = st.tabs([
    "Skill Match",
    "Embeddings",
    "Ablation",
    "Roadmap"
])


with tab1:
    colA, colB = st.columns(2)

    with colA:
        radar = go.Figure(go.Scatterpolar(
            r=[v * 100 for v in similarity.values()],
            theta=list(similarity.keys()),
            fill="toself"
        ))
        radar.update_layout(
            polar=dict(radialaxis=dict(range=[0, 100])),
            height=420
        )
        st.plotly_chart(radar, use_container_width=True)

    with colB:
        bar = go.Figure()
        bar.add_bar(
            x=list(matched.keys()),
            y=[v * 100 for v in matched.values()],
            name="Matched"
        )
        bar.add_bar(
            x=list(missing.keys()),
            y=[v * 100 for v in missing.values()],
            name="Missing"
        )
        bar.update_layout(barmode="group", height=420)
        st.plotly_chart(bar, use_container_width=True)

    st.subheader(" Why skills are missing?")
    for skill, score in missing.items():
        with st.expander(f"{skill.upper()} — {round(score*100,2)}%"):
            if score == 0:
                st.write("Not mentioned in resume")
            elif score < 0.4:
                st.write("Mentioned weakly or out of context")
            else:
                st.write("Semantically distant usage")
            st.write("✔ Add real project usage")
            st.write("✔ Mention tools, metrics, outcomes")


with tab2:
    st.markdown("##  Semantic Skill Space")
    st.caption(
        "Each point is a skill. Distance represents semantic similarity "
        "(closer = used in similar context)."
    )

    skills = list(similarity.keys())
    embeddings = matcher.embed(skills)

    reducer = umap.UMAP(
        n_neighbors=max(2, min(6, len(skills) - 1)),
        min_dist=0.25,
        metric="cosine",
        random_state=42
    )
    coords = reducer.fit_transform(embeddings)

    n_clusters = min(4, len(skills))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    palette = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]

    fig = go.Figure()

    for cid in range(n_clusters):
        idx = [i for i, l in enumerate(labels) if l == cid]
        fig.add_trace(go.Scatter(
            x=coords[idx, 0],
            y=coords[idx, 1],
            mode="markers+text",
            text=[skills[i] for i in idx],
            textposition="top center",
            name=f"Cluster {cid}",
            marker=dict(
                size=14,
                color=palette[cid],
                line=dict(width=1, color="white")
            ),
            hovertemplate="<b>%{text}</b><br>Semantic Group<extra></extra>"
        ))

    fig.update_layout(
        height=520,
        plot_bgcolor="#fafafa",
        paper_bgcolor="#fafafa",
        margin=dict(l=20, r=20, t=20, b=20),
        legend_title="Skill Groups",
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False)
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader(" Auto-Discovered Skill Groups")
    clusters = {}
    for s, l in zip(skills, labels):
        clusters.setdefault(l, []).append(s)

    for l, items in clusters.items():
        st.markdown(
            f"""
            <div class="card" style="border-left:6px solid {palette[l]}">
            <b>Cluster {l}</b><br>
            {", ".join(items)}
            </div>
            """,
            unsafe_allow_html=True
        )

with tab3:
    st.subheader("Model Comparison (Ablation Study)")

    delta = final_score - skill_score
    better_model = "Full Model" if delta >= 0 else "Skills-only"

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="card">
                <h4> Full Model</h4>
                <p style="color: #666;">Section-aware semantic scoring</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.metric(
            label="Score",
            value=f"{final_score:.2f}%",
            delta=f"{delta:+.2f}%" if delta >= 0 else None
        )
        st.progress(min(final_score / 100, 1.0))

    with c2:
        st.markdown(
            """
            <div class="card">
                <h4> Skills-only Model</h4>
                <p style="color: #666;">No section awareness</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.metric(
            label="Score",
            value=f"{skill_score:.2f}%",
            delta=f"{-delta:+.2f}%" if delta < 0 else None
        )
        st.progress(min(skill_score / 100, 1.0))

    # Explanation box (this matters for recruiters & reviewers)
    if delta >= 0:
        st.success(
            f" Removing section awareness reduces performance by **{abs(delta):.2f}%**. "
            f"This shows structural context significantly improves resume–JD matching."
        )
    else:
        st.warning(
            f" Skills-only scoring outperforms the full model by **{abs(delta):.2f}%**. "
            f"This may indicate weak section extraction or noisy resume structure."
        )


with tab4:
    st.subheader("Personalized Improvement Roadmap")

    roadmap = generate_roadmap(list(missing.keys()), final_score)

    if not roadmap:
        st.success(" You are interview-ready. No critical skill gaps detected.")
    else:
        st.caption("Prioritized recommendations based on missing skills and score impact")

        for step in roadmap:
            priority_color = {
                "high": "#ff6b6b",
                "medium": "#f4a261",
                "low": "#2a9d8f"
            }.get(step["priority"].lower(), "#999")

            st.markdown(
                f"""
                <div class="card" style="border-left: 5px solid {priority_color};">
                    <h4 style="margin-bottom: 4px;">🔹 {step['skill'].title()}</h4>
                    <p style="margin: 6px 0; color: #444;">
                        {step['action']}
                    </p>
                    <span style="
                        background: {priority_color};
                        color: white;
                        padding: 4px 10px;
                        border-radius: 12px;
                        font-size: 12px;
                    ">
                        Priority: {step['priority'].upper()}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

