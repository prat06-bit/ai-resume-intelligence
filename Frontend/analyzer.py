import streamlit as st
import plotly.graph_objects as go
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

from backend.skill_extractor import extract_skills, load_skills
from backend.semantic_matcher import SemanticMatcher
from backend.scoring import compute_score, role_weighted_score
from backend.roadmap import generate_roadmap



st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("AI Resume Analyzer")
st.caption("Semantic ML • Explainable ATS • Embedding Analysis")

st.markdown(
    """
    <style>
    .stMetric { text-align: center; }
    .card {
        padding: 16px;
        border-radius: 14px;
        background-color: #f7f7f7;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


role = st.selectbox(
    "Target Role",
    [
        "software_engineer", "backend_engineer", "frontend_engineer",
        "fullstack_engineer", "ml_engineer", "data_scientist",
        "data_analyst", "devops_engineer", "cloud_engineer",
        "student", "intern"
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
            return "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())
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
    similarity = {s: 0.0 for s in jd_skills}

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


m1, m2, m3 = st.columns(3)
m1.metric("Final Match Score", f"{final_score:.1f}%")
m2.metric("Matched Skills", len(matched))
m3.metric("Missing Skills", len(missing))
st.caption(f"Confidence ±{confidence:.1f}%")


tab1, tab2, tab3, tab4 = st.tabs([
    "Skill Match", "Embeddings", "Ablation", "Roadmap"
])


with tab1:
    colA, colB = st.columns(2)

    
    radar = go.Figure(go.Scatterpolar(
        r=[v * 100 for v in similarity.values()],
        theta=list(similarity.keys()),
        fill="toself"
    ))
    radar.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100])),
        height=420
    )
    colA.plotly_chart(radar, use_container_width=True)

    
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
    bar.update_layout(
        barmode="group",
        height=420
    )
    colB.plotly_chart(bar, use_container_width=True)

    
    st.subheader("Why skills are missing & how to improve")

    for skill, score in missing.items():
        with st.expander(f"{skill.upper()} — {round(score*100,1)}%"):
            if score == 0:
                st.write(" Not mentioned anywhere in the resume.")
            elif score < 0.4:
                st.write(" Mentioned weakly or without concrete usage.")
            else:
                st.write(" Used in a context semantically distant from the job role.")

            st.markdown("""
            **How to improve**
            - Add real project usage
            - Mention tools, metrics, and outcomes
            - Show *how* the skill was applied
            """)


with tab2:
    st.subheader("Semantic Skill Space")

    animate = st.checkbox("Animate PCA → UMAP transition", value=True)
    show_density = st.checkbox("Show skill density contours", value=True)

    skills = list(similarity.keys())

    
    if len(skills) < 3:
        st.warning("Not enough skills for semantic projection (need ≥ 3).")
        st.stop()

    
    raw_embeddings = matcher.embed(skills)
    embeddings = np.array(raw_embeddings, dtype=np.float32)

    if not np.isfinite(embeddings).all():
        st.error("Embedding instability detected. Falling back to PCA.")
        HAS_UMAP_SAFE = False
    else:
        HAS_UMAP_SAFE = HAS_UMAP

    
    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(embeddings)

    
    if HAS_UMAP_SAFE:
        try:
            reducer = umap.UMAP(
                n_neighbors=min(8, len(skills) - 1),
                min_dist=0.25,
                metric="cosine",
                random_state=42
            )
            coords_umap = reducer.fit_transform(embeddings)
        except Exception:
            st.warning("UMAP failed numerically. Using PCA instead.")
            coords_umap = coords_pca.copy()
            animate = False
    else:
        coords_umap = coords_pca.copy()
        animate = False

    
    n_clusters = min(4, len(skills))
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(embeddings)

    
    fig = go.Figure()

    for cid in range(n_clusters):
        idx = np.where(labels == cid)[0]
        fig.add_trace(go.Scatter(
            x=coords_pca[idx, 0],
            y=coords_pca[idx, 1],
            mode="markers+text",
            text=[skills[i] for i in idx],
            textposition="top center",
            marker=dict(size=14),
            name=f"Cluster {cid}"
        ))

    
    if show_density:
        xy = np.vstack([coords_pca[:, 0], coords_pca[:, 1]])
        kde = gaussian_kde(xy)

        xgrid, ygrid = np.mgrid[
            coords_pca[:, 0].min():coords_pca[:, 0].max():100j,
            coords_pca[:, 1].min():coords_pca[:, 1].max():100j
        ]

        z = kde(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)

        fig.add_trace(go.Contour(
            x=xgrid[:, 0],
            y=ygrid[0],
            z=z,
            opacity=0.25,
            showscale=False
        ))

    if animate:
        fig.frames = [
            go.Frame(
                data=[go.Scatter(
                    x=coords_umap[:, 0],
                    y=coords_umap[:, 1]
                )],
                name="UMAP"
            )
        ]

        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(
                    label="▶ Morph PCA → UMAP",
                    method="animate",
                    args=[["UMAP"], {"frame": {"duration": 1200, "redraw": True}}]
                )]
            )]
        )

    fig.update_layout(
        height=540,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)


with tab3:
    delta = final_score - skill_score
    c1, c2 = st.columns(2)

    c1.metric("Full Model", f"{final_score:.2f}%", f"{delta:+.2f}%")
    c2.metric("Skills Only", f"{skill_score:.2f}%", f"{-delta:+.2f}%")

    if delta >= 0:
        st.success("Section-aware semantic scoring improves alignment.")
    else:
        st.warning("Skills-only scoring currently performs better.")


with tab4:
    st.subheader("Personalized Improvement Roadmap")
    roadmap = generate_roadmap(list(missing.keys()), final_score)

    if not roadmap:
        st.success("You are interview-ready.")
    else:
        for step in roadmap:
            st.markdown(
                f"""
                <div class="card">
                    <b>{step['skill'].title()}</b><br>
                    {step['action']}<br>
                    <small>Priority: {step['priority'].upper()}</small>
                </div>
                """,
                unsafe_allow_html=True
            )
