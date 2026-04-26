import os
import sys
import warnings
import importlib.util

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from ML_Models.roadmap import generate_roadmap
from ML_Models.semantic_matcher import SemanticMatcher, compute_score, role_weighted_score
from ML_Models.skill_extractor import extract_skills, load_skills


ROLE_OPTIONS = [
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
    "intern",
]

SKILL_SPACE_COLORS = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
    "#000000",
]


def _read_resume_file(uploaded_file) -> str:
    if uploaded_file.type == "application/pdf":
        import pdfplumber

        with pdfplumber.open(uploaded_file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    return uploaded_file.read().decode("utf-8", errors="replace")


def _fit_skill_clusters(embeddings: np.ndarray) -> tuple[np.ndarray, int]:
    n_samples = len(embeddings)
    if n_samples < 3:
        return np.zeros(n_samples, dtype=int), 1

    best_labels = None
    best_score = -np.inf
    upper_k = min(6, n_samples - 1)

    for k in range(2, upper_k + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embeddings)
        if len(np.unique(labels)) < 2:
            continue
        try:
            score = silhouette_score(embeddings, labels, metric="cosine")
        except ValueError:
            score = -np.inf
        if score > best_score:
            best_score = score
            best_labels = labels

    if best_labels is None:
        return np.zeros(n_samples, dtype=int), 1
    return best_labels, int(len(np.unique(best_labels)))


def _cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    sim_matrix = cosine_similarity(embeddings)
    dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    np.fill_diagonal(dist_matrix, 0.0)
    return dist_matrix


def _project_skill_embeddings(
    embeddings: np.ndarray,
    projection: str,
    n_components: int,
) -> np.ndarray:
    n_samples, n_features = embeddings.shape

    if projection == "UMAP":
        if n_samples <= n_components:
            raise ValueError(
                f"UMAP {n_components}D needs at least {n_components + 1} skills; found {n_samples}."
            )
        from umap import UMAP

        reducer = UMAP(
            n_components=n_components,
            n_neighbors=min(max(2, n_samples // 2), n_samples - 1),
            min_dist=0.08,
            metric="cosine",
            init="random",
            random_state=42,
        )
        coords = reducer.fit_transform(embeddings)
    elif projection == "PCA":
        n_safe = min(n_components, n_samples, n_features)
        coords = PCA(n_components=n_safe, random_state=42).fit_transform(embeddings)
    else:
        from sklearn.manifold import MDS

        coords = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=42,
            n_init=4,
        ).fit_transform(_cosine_distance_matrix(embeddings))

    coords = np.asarray(coords, dtype=float)
    if coords.shape[1] < n_components:
        padding = np.zeros((coords.shape[0], n_components - coords.shape[1]))
        coords = np.hstack([coords, padding])
    if not np.isfinite(coords).all():
        raise ValueError(f"{projection} produced non-finite coordinates.")
    return coords


def _skill_source(skill: str, resume_set: set[str], jd_set: set[str]) -> str:
    if skill in resume_set and skill in jd_set:
        return "Resume + JD"
    if skill in resume_set:
        return "Resume"
    return "JD"


def _build_skill_space_frame(
    skills: list[str],
    coords: np.ndarray,
    cluster_ids: np.ndarray,
    resume_skills_set: set[str],
    jd_skills_set: set[str],
    similarity: dict,
) -> pd.DataFrame:
    rows = []
    for idx, skill in enumerate(skills):
        match_score = similarity.get(skill)
        rows.append({
            "skill": skill,
            "label": skill.replace("_", " ").title(),
            "source": _skill_source(skill, resume_skills_set, jd_skills_set),
            "cluster": int(cluster_ids[idx]) + 1,
            "x": float(coords[idx, 0]),
            "y": float(coords[idx, 1]),
            "z": float(coords[idx, 2]) if coords.shape[1] > 2 else 0.0,
            "jd_similarity": None if match_score is None else round(match_score * 100, 1),
        })
    df = pd.DataFrame(rows)
    df["jd_similarity_display"] = df["jd_similarity"].apply(
        lambda value: "Resume-only" if pd.isna(value) else f"{value:.1f}%"
    )
    return df


def _plotly_skill_space(df: pd.DataFrame, projection: str, dimensions: str) -> go.Figure:
    df = df.copy()
    df["cluster_label"] = "Cluster " + df["cluster"].astype(str)
    symbol_map = {"Resume": "circle", "JD": "diamond", "Resume + JD": "square"}

    common_args = dict(
        data_frame=df,
        color="cluster_label",
        symbol="source",
        symbol_map=symbol_map,
        color_discrete_sequence=SKILL_SPACE_COLORS,
        text="label",
        hover_name="label",
        hover_data={
            "source": True,
            "cluster_label": True,
            "jd_similarity_display": True,
            "x": ":.3f",
            "y": ":.3f",
            "z": ":.3f",
            "label": False,
            "cluster": False,
        },
    )

    if dimensions == "3D":
        fig = px.scatter_3d(
            x="x",
            y="y",
            z="z",
            **common_args,
        )
    else:
        fig = px.scatter(
            x="x",
            y="y",
            **common_args,
        )

    fig.update_layout(
        title=f"{projection} - {dimensions} skill clusters",
        template="plotly_white",
        height=680 if dimensions == "3D" else 620,
        margin=dict(l=10, r=10, t=60, b=20),
        legend_title_text="Cluster / source",
        uirevision="skill-space",
    )
    fig.update_traces(
        mode="markers+text",
        textposition="top center",
        marker=dict(size=14, opacity=0.92, line=dict(width=1.2, color="#111827")),
        selector=dict(type="scatter"),
    )
    fig.update_traces(
        mode="markers+text",
        textposition="top center",
        marker=dict(size=7, opacity=0.92, line=dict(width=0.6, color="#111827")),
        selector=dict(type="scatter3d"),
    )
    if dimensions == "3D":
        fig.update_layout(scene=dict(
            xaxis_title="Projection 1",
            yaxis_title="Projection 2",
            zaxis_title="Projection 3",
        ))
    else:
        fig.update_xaxes(title_text="Projection 1", zeroline=True, showgrid=True)
        fig.update_yaxes(title_text="Projection 2", zeroline=True, showgrid=True)
    return fig


def _matplotlib_skill_space(df: pd.DataFrame, projection: str, dimensions: str):
    import matplotlib.pyplot as plt

    marker_map = {"Resume": "o", "JD": "D", "Resume + JD": "s"}
    fig = plt.figure(figsize=(11, 7))
    is_3d = dimensions == "3D"
    ax = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)

    for _, row in df.iterrows():
        color = SKILL_SPACE_COLORS[(int(row["cluster"]) - 1) % len(SKILL_SPACE_COLORS)]
        marker = marker_map[row["source"]]
        if is_3d:
            ax.scatter(row["x"], row["y"], row["z"], color=color, s=90, marker=marker)
        else:
            ax.scatter(row["x"], row["y"], color=color, s=140, marker=marker, edgecolors="#111827")
            ax.text(row["x"], row["y"], row["label"], fontsize=9, ha="center", va="bottom")

    ax.set_title(f"{projection} - {dimensions} skill clusters")
    ax.set_xlabel("Projection 1")
    ax.set_ylabel("Projection 2")
    if is_3d:
        ax.set_zlabel("Projection 3")
    else:
        x_pad = max((df["x"].max() - df["x"].min()) * 0.08, 0.2)
        y_pad = max((df["y"].max() - df["y"].min()) * 0.08, 0.2)
        ax.set_xlim(df["x"].min() - x_pad, df["x"].max() + x_pad)
        ax.set_ylim(df["y"].min() - y_pad, df["y"].max() + y_pad)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _roadmap_context_response(user_input: str, roadmap: list[dict]) -> str:
    if not roadmap:
        return "Your roadmap has no missing-skill steps right now, so focus on strengthening measurable project impact."

    ranked = sorted(
        roadmap,
        key=lambda step: {"high": 0, "medium": 1, "low": 2}.get(step.get("priority"), 1),
    )
    focus = ranked[:3]
    focus_text = ", ".join(step.get("skill", "").replace("_", " ") for step in focus)
    if user_input.strip():
        return f"Based on the generated roadmap, focus first on {focus_text}. Use the action and why fields below each roadmap card as the source of truth for your next resume update."
    return f"Start with {focus_text}; these are the highest-priority gaps in this analysis."


def _chat_with_roadmap(prompt: str, roadmap: list[dict]) -> str:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        return _roadmap_context_response(prompt, roadmap)

    roadmap_summary = "\n".join(
        f"- {step.get('skill', '')} ({step.get('priority', '')}): "
        f"{step.get('action', '')} Why: {step.get('why', '')}"
        for step in roadmap
    )
    try:
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct"),
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a concise career coach. Ground every answer in this roadmap. "
                            "Do not invent skills, employers, projects, courses, or metrics.\n\n"
                            f"ROADMAP:\n{roadmap_summary}"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 220,
            },
            timeout=20,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return _roadmap_context_response(prompt, roadmap)


st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("AI Resume Analyzer")
st.caption("Semantic ML - Explainable ATS - Embedding Analysis")
st.markdown("<style>.stMetric{text-align:center;}</style>", unsafe_allow_html=True)

role = st.selectbox("Target Role", ROLE_OPTIONS)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Resume")
    resume_file = st.file_uploader("Upload Resume (TXT / PDF)", type=["txt", "pdf"], key="resume_file_up")
    resume_text = st.text_area("Or paste resume text", height=200, key="resume_input")

with col2:
    st.subheader("Job Description")
    jd_file = st.file_uploader("Upload JD (TXT / PDF)", type=["txt", "pdf"], key="jd_file_up")
    jd_text = st.text_area("Or paste Job Description", height=200, key="jd_input")

# Process resume file if uploaded
if resume_file:
    try:
        resume_text = _read_resume_file(resume_file)
    except Exception as e:
        st.error(f"Resume file read error: {e}")
        st.stop()

# Process JD file if uploaded
if jd_file:
    try:
        jd_text = _read_resume_file(jd_file)
    except Exception as e:
        st.error(f"JD file read error: {e}")
        st.stop()

# Validate inputs
if not jd_text or len(jd_text.strip()) < 50:
    st.warning("Job description required (min 50 characters). Upload or paste text.")
    st.stop()

if not resume_text or len(resume_text.strip()) < 50:
    st.warning("Resume required (min 50 characters). Upload or paste text.")
    st.stop()

matcher = SemanticMatcher()
try:
    skills_db = load_skills()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

resume_skills = sorted(extract_skills(resume_text, skills_db))
jd_skills = sorted(extract_skills(jd_text, skills_db))

if not resume_skills and not jd_skills:
    st.error("No skills detected. Ensure resume and JD contain technical terms.")
    st.stop()
if not resume_skills:
    st.warning("No skills found in resume; results may be incomplete.")
if not jd_skills:
    st.warning("No skills found in JD; results may be incomplete.")

try:
    similarity = matcher.match_skills(resume_skills, jd_skills) or {skill: 0.0 for skill in jd_skills}
    section_embeddings = matcher.embed_sections(resume_text)
    jd_emb = matcher.embed([jd_text])[0]
except RuntimeError as e:
    st.error(f"Embedding model unavailable: {e}")
    st.stop()

section_similarities = {
    section: float(cosine_similarity([embedding], [jd_emb])[0][0])
    for section, embedding in section_embeddings.items()
    if embedding is not None
}

skill_score, confidence = compute_score(similarity)
section_score = role_weighted_score(section_similarities, role)
matched = {skill: score for skill, score in similarity.items() if score >= 0.30}
missing = {skill: score for skill, score in similarity.items() if score < 0.30}
matched_count = len(matched)
total_jd = len(similarity) or 1
coverage_pct = (matched_count / total_jd) * 100
final_score = round((skill_score * 0.50) + (section_score * 0.30) + (coverage_pct * 0.20), 1)

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Final Match Score", f"{final_score:.1f}%")
metric2.metric("Matched Skills", matched_count)
metric3.metric("Missing Skills", len(missing))
st.caption(f"Confidence: {confidence} - Coverage: {matched_count}/{total_jd} JD skills")

try:
    roadmap = generate_roadmap(
        missing_skills=list(missing.keys()),
        score=final_score,
        jd_text=jd_text,
        resume_text=resume_text,
    )
except Exception as e:
    st.warning(f"Roadmap generation failed: {e}")
    roadmap = []

tab1, tab2, tab3, tab4 = st.tabs([
    "Skill Match",
    "Embeddings",
    "Score Breakdown",
    "Roadmap",
])

with tab1:
    col_a, col_b = st.columns(2)

    radar = go.Figure(go.Scatterpolar(
        r=[score * 100 for score in similarity.values()],
        theta=[skill.replace("_", " ").title() for skill in similarity.keys()],
        fill="toself",
    ))
    radar.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100])),
        height=420,
        showlegend=False,
        template="plotly_white",
    )
    col_a.subheader("Overall Skill Alignment")
    col_a.plotly_chart(radar, use_container_width=True)

    bar = go.Figure()
    bar.add_bar(
        x=[skill.replace("_", " ").title() for skill in matched.keys()] or ["None"],
        y=[score * 100 for score in matched.values()] or [0],
        name="Matched",
        marker_color="#2A9D8F",
    )
    bar.add_bar(
        x=[skill.replace("_", " ").title() for skill in missing.keys()] or ["None"],
        y=[score * 100 for score in missing.values()] or [0],
        name="Weak / Missing",
        marker_color="#E76F51",
    )
    bar.update_layout(
        barmode="group",
        height=420,
        yaxis_title="Semantic coverage (%)",
        template="plotly_white",
    )
    col_b.subheader("JD Skill Coverage")
    col_b.plotly_chart(bar, use_container_width=True)

    st.subheader("Skill Gaps")
    if not missing:
        st.success("All detected JD skills are covered at the current match threshold.")
    else:
        gap_rows = [
            {
                "Skill": skill.replace("_", " ").title(),
                "Coverage": f"{score * 100:.1f}%",
                "Priority": "High" if score < 0.20 else "Medium" if score < 0.40 else "Low",
            }
            for skill, score in sorted(missing.items(), key=lambda item: item[1])
        ]
        st.dataframe(pd.DataFrame(gap_rows), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Semantic Skill Space")
    st.caption("Circles = Resume | Diamonds = JD | Squares = Both")
 
    all_skills = sorted(set(resume_skills) | set(jd_skills))
    if len(all_skills) < 3:
        st.warning("Need ≥3 skills.")
    else:
        c1, c2, c3 = st.columns(3)
        proj = c1.radio("Projection", ["MDS", "PCA", "UMAP"], horizontal=True)
        dim = c2.radio("Dimensions", ["2D", "3D"], horizontal=True)
        n_comp = 3 if dim == "3D" else 2
 
        try:
            # Get embeddings
            emb = np.array(matcher.embed(all_skills), dtype=np.float32)
            st.write(f"✓ Got {len(all_skills)} embeddings, shape: {emb.shape}")
            
            # Cluster
            k = min(5, max(2, len(all_skills) // 3))
            km = KMeans(n_clusters=k, random_state=42, n_init=5)
            clusters = km.fit_predict(emb)
            
            # Project
            if proj == "MDS":
                from sklearn.manifold import MDS
                dist = 1 - cosine_similarity(emb)
                coords = MDS(n_components=n_comp, dissimilarity='precomputed', random_state=42).fit_transform(dist)
            elif proj == "PCA":
                n_safe = min(n_comp, emb.shape[0], emb.shape[1])
                coords = PCA(n_components=n_safe, random_state=42).fit_transform(emb)
            else:  # UMAP
                try:
                    import umap.umap_ as umap_lib
                    coords = umap_lib.UMAP(n_components=n_comp, n_neighbors=min(15, len(all_skills)-1), random_state=42).fit_transform(emb)
                except:
                    dist = 1 - cosine_similarity(emb)
                    coords = MDS(n_components=n_comp, dissimilarity='precomputed', random_state=42).fit_transform(dist)
            
            if coords.shape[1] < 3:
                coords = np.hstack([coords, np.zeros((len(coords), 3-coords.shape[1]))])
            
            st.write(f"✓ Projected to {coords.shape}, k={k}")
 
            # MATPLOTLIB ONLY
            colors = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#9B5DE5"]
            resume_set = set(resume_skills)
            jd_set = set(jd_skills)
            marker_map = {"circle": "o", "diamond": "D", "square": "s"}
            
            fig = plt.figure(figsize=(12, 8))
            
            if dim == "2D":
                ax = fig.add_subplot(111)
                for i, skill in enumerate(all_skills):
                    in_r = skill in resume_set
                    in_j = skill in jd_set
                    
                    if in_r and in_j:
                        marker = "s"
                    elif in_r:
                        marker = "o"
                    else:
                        marker = "D"
                    
                    color = colors[clusters[i] % len(colors)]
                    ax.scatter(coords[i, 0], coords[i, 1], 
                              c=color, s=250, marker=marker, 
                              alpha=0.8, edgecolors="white", linewidth=2, zorder=10)
                    ax.text(coords[i, 0], coords[i, 1] + 0.15, skill, 
                           ha='center', fontsize=10, weight='bold', zorder=11)
                
                ax.set_xlabel("Projection 1", fontsize=12)
                ax.set_ylabel("Projection 2", fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_facecolor("#FAFAFA")
            else:  # 3D
                ax = fig.add_subplot(111, projection='3d')
                for i, skill in enumerate(all_skills):
                    in_r = skill in resume_set
                    in_j = skill in jd_set
                    
                    if in_r and in_j:
                        marker = "s"
                    elif in_r:
                        marker = "o"
                    else:
                        marker = "D"
                    
                    color = colors[clusters[i] % len(colors)]
                    ax.scatter(coords[i, 0], coords[i, 1], coords[i, 2],
                              c=color, s=150, marker=marker, 
                              alpha=0.8, edgecolors="white", linewidth=1, zorder=10)
                
                ax.set_xlabel("Projection 1", fontsize=10)
                ax.set_ylabel("Projection 2", fontsize=10)
                ax.set_zlabel("Projection 3", fontsize=10)
            
            fig.suptitle(f"{proj} — {dim} Skill Clusters (k={k})", fontsize=14, weight='bold')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            # Data table
            with st.expander("📊 Data"):
                df_data = pd.DataFrame({
                    "Skill": all_skills,
                    "Source": ["Resume" if s in resume_set and s not in jd_set else "JD" if s in jd_set and s not in resume_set else "Both" for s in all_skills],
                    "Cluster": [f"C{c}" for c in clusters],
                    "Similarity": [f"{similarity.get(s, 0)*100:.0f}%" for s in all_skills],
                })
                st.dataframe(df_data, use_container_width=True, hide_index=True)
 
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())
            
with tab3:
    st.subheader("Score Breakdown")
    score_cols = st.columns(3)
    score_cols[0].metric("Skill Score", f"{skill_score:.1f}%")
    score_cols[1].metric("Section Alignment", f"{section_score:.1f}%")
    score_cols[2].metric("Skill Coverage", f"{coverage_pct:.1f}%")

    formula_df = pd.DataFrame([
        {
            "Component": "Skill score",
            "Value": f"{skill_score:.1f}%",
            "Weight": "50%",
            "Points": round(skill_score * 0.50, 1),
        },
        {
            "Component": "Section alignment",
            "Value": f"{section_score:.1f}%",
            "Weight": "30%",
            "Points": round(section_score * 0.30, 1),
        },
        {
            "Component": "JD skill coverage",
            "Value": f"{coverage_pct:.1f}%",
            "Weight": "20%",
            "Points": round(coverage_pct * 0.20, 1),
        },
    ])
    st.dataframe(formula_df, use_container_width=True, hide_index=True)

    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=final_score,
        delta={"reference": 70, "increasing": {"color": "green"}},
        title={"text": "ATS Match Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#457B9D"},
            "steps": [
                {"range": [0, 40], "color": "#fde8e8"},
                {"range": [40, 70], "color": "#fef3cd"},
                {"range": [70, 100], "color": "#d4edda"},
            ],
            "threshold": {"line": {"color": "#16a34a", "width": 3}, "value": 70},
        },
    ))
    gauge.update_layout(height=280)
    st.plotly_chart(gauge, use_container_width=True)

    st.divider()
    st.markdown("### JD Skills Needing Attention")
    if missing:
        missing_sorted = sorted(missing.items(), key=lambda item: item[1])[:10]
        kw_names = [skill.replace("_", " ").title() for skill, _ in missing_sorted]
        kw_scores = [round(score * 100, 1) for _, score in missing_sorted]
        colors = ["#e5533d" if score < 20 else "#f59e0b" if score < 40 else "#457B9D" for score in kw_scores]
        fig_kw = go.Figure(go.Bar(
            x=kw_scores,
            y=kw_names,
            orientation="h",
            marker_color=colors,
            text=[f"{score}%" for score in kw_scores],
            textposition="outside",
        ))
        fig_kw.update_layout(
            height=max(250, len(kw_names) * 38),
            xaxis=dict(title="Current semantic coverage %", range=[0, 110]),
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
            margin=dict(l=10, r=60, t=10, b=30),
        )
        st.plotly_chart(fig_kw, use_container_width=True)
    else:
        st.success("No weak JD skills at the current threshold.")

    st.divider()
    st.markdown("### Roadmap-Backed Resume Actions")
    if roadmap:
        action_rows = [
            {
                "Skill": step.get("skill", "").replace("_", " ").title(),
                "Priority": step.get("priority", "").title(),
                "Action": step.get("action", ""),
                "Why": step.get("why", ""),
            }
            for step in roadmap
        ]
        st.dataframe(pd.DataFrame(action_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No roadmap actions were generated for this analysis.")

    st.divider()
    st.markdown("### Resume Signals")
    signal_rows = [
        {
            "Signal": "Resume length",
            "Status": "OK" if len(resume_text) > 500 else "Needs detail",
            "Evidence": f"{len(resume_text)} characters",
        },
        {
            "Signal": "JD skill coverage",
            "Status": "OK" if matched_count >= total_jd * 0.5 else "Needs work",
            "Evidence": f"{matched_count}/{total_jd} matched",
        },
        {
            "Signal": "Quantified achievements",
            "Status": "Detected" if any(char.isdigit() for char in resume_text) else "Missing",
            "Evidence": "Numbers found" if any(char.isdigit() for char in resume_text) else "No numbers found",
        },
        {
            "Signal": "Portfolio or code link",
            "Status": "Detected" if any(word in resume_text.lower() for word in ["github", "gitlab", "portfolio", "linkedin"]) else "Missing",
            "Evidence": "Link keyword found" if any(word in resume_text.lower() for word in ["github", "gitlab", "portfolio", "linkedin"]) else "No portfolio keyword found",
        },
    ]
    st.dataframe(pd.DataFrame(signal_rows), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Personalized Improvement Plan")
    st.caption("Generated from the missing JD skills, the resume text, and the job description.")

    if not roadmap:
        st.success("No missing-skill roadmap items were generated.")
    else:
        for index, step in enumerate(roadmap, 1):
            priority = step.get("priority", "medium")
            color = "#e5533d" if priority == "high" else "#f59e0b" if priority == "medium" else "#16a34a"
            skill_label = step.get("skill", "").replace("_", " ").title()
            st.markdown(
                f"""
                <div style="padding:16px;border-radius:8px;background:#ffffff;
                            margin-bottom:12px;border-left:6px solid {color};
                            box-shadow:0 2px 10px rgba(0,0,0,0.04);">
                    <div style="display:flex;justify-content:space-between;gap:12px;">
                        <b>Step {index}: {skill_label}</b>
                        <span style="font-size:12px;color:{color};font-weight:700;">{priority.title()}</span>
                    </div>
                    <div style="margin-top:8px;">{step.get("action", "")}</div>
                    <div style="margin-top:8px;font-size:12px;color:#6b7280;">{step.get("why", "")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("Ask About Your Roadmap")

    if "roadmap_chat" not in st.session_state:
        st.session_state.roadmap_chat = []

    for message in st.session_state.roadmap_chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about priority, sequencing, or resume edits..."):
        st.session_state.roadmap_chat.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = _chat_with_roadmap(prompt, roadmap)
            st.markdown(reply)
        st.session_state.roadmap_chat.append({"role": "assistant", "content": reply})