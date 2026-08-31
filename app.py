import streamlit as st
import pickle, json, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import pairwise_distances
import umap

st.set_page_config(page_title="ZSL / FSL / Metric Learning Demo", layout="wide")

BAR_WIDTH = 0.35

@st.cache_data
def load_all():
    with open("streamlit_export.pkl", "rb") as f:
        exp = pickle.load(f)
    with open("fewshot_results_v2.json") as f:
        fs = json.load(f)
    with open("zeroshot_results.json") as f:
        zsl = json.load(f)
    return exp, fs, zsl

data, fs, zsl = load_all()
CLASS_GROUPS = data["class_groups"]
ALL_COLORS = px.colors.qualitative.Bold

def color_lookup(class_list):
    return {c: ALL_COLORS[i % len(ALL_COLORS)] for i, c in enumerate(class_list)}

def labeled_bar(x, y, name, color, width=BAR_WIDTH, fmt=".0%"):
    return go.Bar(x=x, y=y, name=name, marker_color=color, width=width,
                   text=[f"{v:{fmt}}" for v in y], textposition="outside",
                   textfont=dict(size=13, color="black"))

st.title("🐾 Zero-Shot vs Few-Shot vs Normal Classification — Live Demo")
st.caption("Same backbone, same dataset, same embedding space — three different ways of handling new classes.")

page = st.sidebar.radio("Choose a section", [
    "1. Overview & Unified Metrics",
    "2. Metric Learning: Before vs After",
    "3. Few-Shot Learning",
    "4. Zero-Shot Learning",
    "5. Live Classification Demo",
])

if page.startswith("1"):
    st.header("Unified Model Comparison")
    st.warning(
        "⚠️ Read this before the bars below: these three models are NOT solving the same task. "
        "Normal is tested on its 10 known classes with full training data. Few-shot is tested on 5 "
        "minority classes it never trained on, given only a few examples at test time. Zero-shot is "
        "tested on 5 unseen classes with zero training images, using only attribute descriptions. "
        "The fair comparison is each blue bar against its OWN gray chance-level bar."
    )
    summary_df = pd.DataFrame(data["summary_table"])
    fig = go.Figure()
    fig.add_trace(labeled_bar(summary_df["Model"], summary_df["Accuracy"], "Achieved accuracy", "#4C72B0"))
    fig.add_trace(labeled_bar(summary_df["Model"], summary_df["Chance level"], "Random chance", "#BBBBBB"))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.15,
                       yaxis_tickformat=".0%", yaxis_range=[0, 1.15],
                       title="Every model beats its OWN chance-level baseline — same backbone throughout",
                       height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shows: each paradigm's improvement over guessing, given its own difficulty level. "
               "Does NOT show: which model is objectively best — the tasks are not comparable.")
    st.dataframe(summary_df.style.format({
        "Accuracy": "{:.1%}", "Chance level": "{:.1%}", "Improvement over chance (x)": "{:.2f}x"
    }), use_container_width=True)

elif page.startswith("2"):
    st.header("Does Metric Learning Actually Pull Same-Class Points Together?")
    col1, col2 = st.columns(2)
    col1.metric("Silhouette score — BEFORE", f"{data['silhouette_before']:.3f}")
    col2.metric("Silhouette score — AFTER", f"{data['silhouette_after']:.3f}",
                delta=f"{data['silhouette_after']-data['silhouette_before']:+.3f}")
    st.caption("Higher silhouette score = same-class points sit closer together, different-class points sit farther apart.")

    @st.cache_data
    def project_before_after():
        r1 = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
        before_2d = r1.fit_transform(data["before_metric_learning_embeddings"])
        r2 = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
        after_2d = r2.fit_transform(data["majority_test_embeddings"])
        return before_2d, after_2d

    before_2d, after_2d = project_before_after()
    cmap = color_lookup(CLASS_GROUPS["majority"])

    with col1:
        st.subheader("Before (ImageNet backbone only)")
        fig1 = go.Figure()
        for c in CLASS_GROUPS["majority"]:
            idx = [i for i, l in enumerate(data["before_metric_learning_labels"]) if l == c]
            fig1.add_scatter(x=before_2d[idx, 0], y=before_2d[idx, 1], mode="markers",
                              name=c, marker=dict(color=cmap[c], size=6, opacity=0.6))
        fig1.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("After (Prototypical Network training)")
        fig2 = go.Figure()
        for c in CLASS_GROUPS["majority"]:
            idx = [i for i, l in enumerate(data["majority_test_labels"]) if l == c]
            fig2.add_scatter(x=after_2d[idx, 0], y=after_2d[idx, 1], mode="markers",
                              name=c, marker=dict(color=cmap[c], size=6, opacity=0.6))
        fig2.update_layout(height=450)
        st.plotly_chart(fig2, use_container_width=True)

    st.success("Same images, same backbone — only the training objective changed. Tighter, more separated "
               "clusters on the right are the visual signature of metric learning.")

elif page.startswith("3"):
    st.header("Few-Shot Learning on Minority Classes")
    st.write("Backbone trained on majority classes only. Minority classes were never used in training — "
             "we compute new prototypes from a handful of examples at test time.")

    k_vals = list(fs["kshot_sweep"].keys())
    means = [fs["kshot_sweep"][k]["mean"] for k in k_vals]
    stds = [fs["kshot_sweep"][k]["std"] for k in k_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"k={k}" for k in k_vals], y=means, error_y=dict(type="data", array=stds, visible=True),
        marker_color="#55A868", width=BAR_WIDTH,
        text=[f"{m:.0%}" for m in means], textposition="outside", textfont=dict(size=13)
    ))
    fig.add_hline(y=fs["chance_level"], line_dash="dash", line_color="gray",
                  annotation_text=f"Random chance ({fs['chance_level']:.0%})")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1.1], bargap=0.5,
                       title="Accuracy vs. k — mean ± std over 15 random draws", height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Accuracy is stable and above chance across k values; noise is shown honestly as error bars.")

    st.subheader("Does augmentation + bootstrapping help at 1-shot?")
    ab = fs["aug_bootstrap_ablation"]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=["Without aug/bootstrap", "With aug/bootstrap"],
        y=[ab["without"]["mean"], ab["with"]["mean"]],
        error_y=dict(type="data", array=[ab["without"]["std"], ab["with"]["std"]], visible=True),
        marker_color=["#C44E52", "#55A868"], width=BAR_WIDTH,
        text=[f"{ab['without']['mean']:.0%}", f"{ab['with']['mean']:.0%}"],
        textposition="outside", textfont=dict(size=13)
    ))
    fig2.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1.1], bargap=0.5,
                        title="1-shot prototype quality: raw single image vs. augmented + bootstrapped", height=450)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Run at the hardest setting (k=1) since that's where a noisy single-image prototype hurts most. "
               "The benefit shrinks at larger k since the prototype is already averaged over more real images.")

    st.subheader("Normal model FORCED to guess on minority-class images")
    rows = [{"True class": c, "Guess": p} for c, preds in fs["confusion_normal_on_minority"].items() for p in preds]
    conf_df = pd.DataFrame(rows)
    conf_pivot = pd.crosstab(conf_df["True class"], conf_df["Guess"], normalize="index")
    fig3 = px.imshow(conf_pivot, color_continuous_scale="Reds", aspect="auto", text_auto=".0%",
                      labels=dict(color="Fraction"), title="Where does the normal model's guess land?")
    fig3.update_traces(textfont_size=11)
    st.plotly_chart(fig3, use_container_width=True)

elif page.startswith("4"):
    st.header("Zero-Shot Learning: Classifying Animals With ZERO Training Images")
    st.write("These 5 classes contributed no images to training. Prototypes (★) are synthesized purely from "
             "85-dim attribute vectors — the semantic/auxiliary vector for this project.")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Without semantics", "With semantics"],
        y=[zsl["acc_without_semantics"], zsl["acc_with_semantics"]],
        marker_color=["#C44E52", "#4C72B0"], width=BAR_WIDTH,
        text=[f"{zsl['acc_without_semantics']:.0%}", f"{zsl['acc_with_semantics']:.0%}"],
        textposition="outside", textfont=dict(size=13)
    ))
    fig.add_hline(y=zsl["chance_level"], line_dash="dash", line_color="gray",
                  annotation_text=f"Random chance ({zsl['chance_level']:.0%})")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1.1], bargap=0.5,
                       title="Same architecture, same training — only the vector's semantic CONTENT differs", height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Isolates whether real semantic content matters, vs. just having some extra vector to condition on.")

    st.subheader("Synthesized prototypes (★) vs. real unseen-class images (dots)")

    @st.cache_data
    def project_zsl():
        combined = np.vstack([data["unseen_embeddings"], data["zsl_prototypes_real"]])
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
        combined_2d = reducer.fit_transform(combined)
        n = data["unseen_embeddings"].shape[0]
        return combined_2d[:n], combined_2d[n:]

    imgs_2d, protos_2d = project_zsl()
    cmap = color_lookup(CLASS_GROUPS["unseen"])
    fig2 = go.Figure()
    for c in CLASS_GROUPS["unseen"]:
        idx = [i for i, l in enumerate(data["unseen_labels"]) if l == c]
        fig2.add_scatter(x=imgs_2d[idx, 0], y=imgs_2d[idx, 1], mode="markers",
                          name=f"{c} (real images)", marker=dict(color=cmap[c], size=7, opacity=0.5))
    for i, c in enumerate(data["zsl_prototypes_real_order"]):
        fig2.add_scatter(x=[protos_2d[i, 0]], y=[protos_2d[i, 1]], mode="markers+text",
                          marker=dict(color=cmap[c], size=22, symbol="star", line=dict(color="black", width=2)),
                          text=[f"★ {c}"], textposition="top center", textfont=dict(size=12, color="black"),
                          name=f"{c} prototype", showlegend=False)
    fig2.update_layout(height=600)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("★ = placed using ONLY that class's attribute vector, via the trained semantic mapper — "
               "no image of this class was ever used to place it.")

# ============================================================
# PAGE 5 — Live classification demo (the showpiece for presenting live)
# ============================================================
elif page.startswith("5"):
    st.header("🎬 Live Demo: Classify an Unseen Point")
    st.write("Pick a real example and watch which prototype it gets matched to — this is the moment to slow down "
             "and narrate during your presentation.")

    demo_type = st.selectbox("Choose classification scenario:",
                              ["Few-shot: unseen minority-class image", "Zero-shot: unseen class image"])

    if demo_type.startswith("Few-shot"):
        chosen_class = st.selectbox("Pick a minority class:", CLASS_GROUPS["minority"])
        idx_options = [i for i, l in enumerate(data["minority_labels"]) if l == chosen_class]
        chosen_idx = st.selectbox("Pick a test point index:", idx_options)
        query_emb = data["minority_embeddings"][chosen_idx]
        protos = data["fewshot_prototypes_k5"]
        proto_order = data["fewshot_prototypes_k5_order"]
    else:
        chosen_class = st.selectbox("Pick an unseen class:", CLASS_GROUPS["unseen"])
        idx_options = [i for i, l in enumerate(data["unseen_labels"]) if l == chosen_class]
        chosen_idx = st.selectbox("Pick a test point index:", idx_options)
        query_emb = data["unseen_embeddings"][chosen_idx]
        protos = data["zsl_prototypes_real"]
        proto_order = data["zsl_prototypes_real_order"]

    dists = pairwise_distances(query_emb.reshape(1, -1), protos)[0]
    pred_idx = np.argmin(dists)
    pred_class = proto_order[pred_idx]
    correct = pred_class == chosen_class

    if st.button("▶ Classify this point"):
        placeholder = st.empty()
        for pct in [20, 45, 70, 100]:
            placeholder.progress(pct, text="Computing distance to each prototype...")
            time.sleep(0.25)
        placeholder.empty()

        st.write(f"**True class:** {chosen_class}")
        st.write(f"**Nearest prototype:** {pred_class}")
        if correct:
            st.success(f"✅ Correctly classified! Distance to correct prototype: {dists[pred_idx]:.3f}")
        else:
            st.error(f"❌ Misclassified. Predicted '{pred_class}' instead of '{chosen_class}'.")

        dist_df = pd.DataFrame({"Class": proto_order, "Distance": dists}).sort_values("Distance")
        fig = px.bar(dist_df, x="Class", y="Distance", color="Distance", color_continuous_scale="RdYlGn_r",
                     title="Distance from this point to every candidate prototype (lower = more likely)")
        st.plotly_chart(fig, use_container_width=True)
