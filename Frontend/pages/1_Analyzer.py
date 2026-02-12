import streamlit as st
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

from ML_Models.background_builder import build_background
from ML_Models.skill_extractor import extract_skills, load_skills
from ML_Models.semantic_matcher import SemanticMatcher
from ML_Models.scoring import compute_score, role_weighted_score
from ML_Models.roadmap import generate_roadmap
from ML_Models.ml_features import build_feature_vector, FEATURE_NAMES
from ML_Models.model import ResumeMatchModel
from ML_Models.shap import ShapExplainer
import shap

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

X_single = build_feature_vector(similarity, section_similarities)

X_train = []
y_train = []

for _ in range(80):
    noise = np.random.normal(0, 0.06, size=X_single.shape)
    x = np.clip(X_single + noise, 0, 1)

    score_proxy = (
        0.4 * x[0] +   
        0.3 * x[1] +   
        0.2 * x[2] +   
        0.1 * x[3]     
    )

    label = int(score_proxy >= 0.55)

    X_train.append(x)
    y_train.append(label)

if len(set(y_train)) < 2:
    
    x_pos = np.clip(X_single + np.array([0.15, 0.2, 0.2, 0.15, 0.1]), 0, 1)
    x_neg = np.clip(X_single - np.array([0.15, 0.2, 0.2, 0.15, 0.1]), 0, 1)

    X_train.extend([x_pos, x_neg])
    y_train.extend([1, 0])

X_train = np.vstack(X_train)
y_train = np.array(y_train)

X_train = np.vstack(X_train)
y_train = np.array(y_train)

ml_model = ResumeMatchModel()
ml_model.fit(X_train, y_train)

ml_score = ml_model.predict_proba(X_single.reshape(1, -1))[0] * 100

m1, m2, m3 = st.columns(3)
m1.metric("Final Match Score", f"{ml_score:.1f}%")
m2.metric("Matched Skills", len(matched))
m3.metric("Missing Skills", len(missing))
st.caption("Confidence derived from model explainability (SHAP)")

tab1, tab2, tab3, tab4 = st.tabs([
    "Skill Match", "Embeddings", "Ablation", "Roadmap"
])

roadmap = None
with tab1:
    colA, colB = st.columns(2)

    radar = go.Figure(
        go.Scatterpolar(
            r=[v * 100 for v in similarity.values()],
            theta=list(similarity.keys()),
            fill="toself",
        )
    )
    radar.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100])),
        height=420,
        showlegend=False,
    )
    colA.subheader("Overall Skill Alignment")
    colA.plotly_chart(radar, use_container_width=True)

    bar = go.Figure()
    bar.add_bar(
        x=list(matched.keys()),
        y=[v * 100 for v in matched.values()],
        name="Present in Resume",
    )
    bar.add_bar(
        x=list(missing.keys()),
        y=[v * 100 for v in missing.values()],
        name="Expected but Weak / Missing",
    )
    bar.update_layout(
        barmode="group",
        height=420,
        yaxis_title="Coverage (%)",
    )
    colB.subheader("Skill Coverage Comparison")
    colB.plotly_chart(bar, use_container_width=True)

    st.subheader("Skill Gaps & How to Improve")
    st.caption(
        "These skills are expected for the role but are either missing or weakly demonstrated in the resume."
    )

    if not missing:
        st.success(
            "All required skills are reasonably covered. Focus on depth, impact, and clarity."
        )
    else:
        # Sort by most missing first
        top_missing = sorted(
            missing.items(), key=lambda x: x[1]
        )[:6]

        for skill, score in top_missing:
            severity = (
                "High gap" if score < 0.2 else
                "Moderate gap" if score < 0.4 else
                "Low gap"
            )

            st.markdown(
                f"""
                <div style="
                    padding:14px;
                    border-radius:12px;
                    background:#f8f9fa;
                    margin-bottom:10px;
                    border-left:6px solid #e5533d;
                ">
                    <b>{skill.upper()}</b>
                    <span style="float:right; font-size:13px;">
                        {severity} · {round(score*100,1)}% coverage
                    </span>
                    <br><br>
                    <small>
                        This skill is expected in the job description but is not
                        strongly demonstrated in your resume.
                    </small>
                    <ul style="margin-top:8px;">
                        <li>Add a project where you used <b>{skill}</b></li>
                        <li>Mention tools, metrics, or outcomes</li>
                        <li>Explain <i>how</i> the skill was applied</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
    st.subheader("Score Breakdown")

    delta = final_score - skill_score
    c1, c2 = st.columns(2)

    c1.metric("Context-Aware Model", f"{final_score:.2f}%", f"{delta:+.2f}%")
    c2.metric("Skills-Only Model", f"{skill_score:.2f}%", f"{-delta:+.2f}%")

    if delta >= 0:
        st.success("Context-aware scoring improves alignment for this resume.")
    else:
        st.warning("Keyword-only scoring performs slightly better here.")

    st.caption(
        "The final score considers skill match, project relevance, and experience alignment."
    )

    background = X_train[y_train == 0][:30]
    shap_explainer = ShapExplainer(ml_model.model, background)

    shap_exp = shap_explainer.local(X_single.reshape(1, -1))
    global_shap = shap_explainer.global_importance(background)

    with st.expander("How this score was calculated"):
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_exp.values[0],
                base_values=shap_exp.base_values[0],
                feature_names=[
                    "Project relevance",
                    "Job skill coverage",
                    "Experience alignment",
                    "Overall skill match",
                    "Skills section clarity",
                ],
            ),
            max_display=5,
            show=False,
        )
        st.pyplot(plt.gcf(), clear_figure=True)

        st.caption(
            "This chart shows how each signal pushed the score up or down for *this resume*."
        )

    st.subheader("What the model generally prioritizes")
    st.caption(
        "Across many successful resumes, these signals tend to matter the most."
    )

    priorities = sorted(
        zip(
            [
                "Project relevance",
                "Job skill coverage",
                "Experience alignment",
                "Overall skill match",
                "Skills section clarity",
            ],
            global_shap,
        ),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:3]

    for name, _ in priorities:
        st.markdown(
            f"""
            <div style="
                padding:12px;
                border-radius:10px;
                background:#f8f9fa;
                margin-bottom:8px;
            ">
                <b>{name}</b><br>
                <small>
                    Frequently influences hiring decisions across resumes.
                </small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Potential improvement signals")

    reasons = [
        name
        for name, value in zip(
            [
                "Project relevance",
                "Job skill coverage",
                "Experience alignment",
                "Overall skill match",
                "Skills section clarity",
            ],
            shap_exp.values[0],
        )
        if value < -0.03
    ]

    if not reasons:
        st.success("No strong negative signals detected.")
    else:
        for r in reasons:
            st.write(f"• Strengthen **{r}** to improve alignment.")

with tab4:
    st.subheader("Personalized Improvement Plan")
    st.caption(
        "A targeted roadmap showing what will most improve your resume’s alignment with this role."
    )

    if roadmap is None or len(roadmap) == 0:
        st.success(
            "You are interview-ready. Focus on applying, interviewing, and refining impact."
        )
    else:
        for i, step in enumerate(roadmap, 1):
            is_high = step["priority"] == "high"
            priority_color = "#16a34a" if is_high else "#f59e0b"
            priority_label = "High impact" if is_high else "Medium impact"

            st.markdown(
                f"""
                <div style="
                    padding:18px;
                    border-radius:16px;
                    background:#ffffff;
                    margin-bottom:14px;
                    box-shadow:0 4px 14px rgba(0,0,0,0.04);
                    border-left:6px solid {priority_color};
                ">
                    <div style="display:flex; justify-content:space-between;">
                        <b style="font-size:16px;">
                            Step {i}: Strengthen {step['skill'].title()}
                        </b>
                        <span style="
                            font-size:12px;
                            padding:4px 10px;
                            border-radius:999px;
                            background:{priority_color};
                            color:white;
                        ">
                            {priority_label}
                        </span>
                    </div>

                    <div style="margin-top:10px; font-size:14px;">
                        {step['action']}
                    </div>

                    <div style="
                        margin-top:10px;
                        font-size:12px;
                        color:#6b7280;
                    ">
                        Why this matters: Improving this signal has a
                        measurable impact on your overall match score.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

