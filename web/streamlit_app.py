"""Streamlit UI for cross-modal satellite image retrieval.

Run:
    pip install -r requirements-web.txt
    streamlit run web/streamlit_app.py

Reuses the project's ``src/`` pipeline and persistent stores. Tabs:
Home / Retrieval / Results / Map / Analytics / Embeddings / Explainability /
Gallery / History.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.interface import build_dataset
from src.evaluation.latency import run_latency_benchmark
from src.pipeline import (
    build_loaders,
    load_best_model,
    modality_channels_from_patches,
    prepare_dataset,
)
from src.retrieval.engine import RetrievalEngine, result_records
from src.retrieval.rerank import build_reranker
from src.utils.config import load_config
from src.utils.embedding_viz import plot_embedding_comparison, plot_embeddings, reduce_embeddings
from src.utils.io import Logger, resolve_device
from src.utils.visualize import render_patch
from src.xai.attention import attention_map
from src.xai.gradcam import gradcam, overlay_saliency

st.set_page_config(page_title="Cross-Modal Satellite Image Retrieval", layout="wide")


# ---------------------------------------------------------------------------
# Pipeline loading (cached once per process)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pipeline() -> dict:
    cfg_path = os.environ.get(
        "RETRIEVAL_CONFIG", os.path.join(ROOT, "configs", "default.yaml")
    )
    cfg = load_config(cfg_path)
    device = resolve_device(cfg["training"]["device"])
    logger = Logger(path=None)
    patches, labels, class_names, stats, transforms, metadata = prepare_dataset(cfg, logger)
    model = load_best_model(
        cfg,
        len(class_names),
        device,
        modality_channels=modality_channels_from_patches(patches, cfg["modalities"]),
    ).to(device)
    _, _, full_ds = build_loaders(
        cfg, patches, labels, transforms, stats=stats, metadata=metadata
    )
    engine = RetrievalEngine(model, full_ds, device)

    n = len(full_ds)
    gallery_ids = np.arange(n // 2, n)
    index_kwargs = {
        "index_type": cfg["retrieval"]["index"].get("type", "flat"),
        "metric": cfg["retrieval"]["index"].get("metric", "cosine"),
    }
    galleries = {
        m: engine.build_gallery(gallery_ids, m, index_kwargs=index_kwargs)
        for m in cfg["modalities"]
    }
    return {
        "cfg": cfg,
        "device": device,
        "patches": patches,
        "labels": labels,
        "class_names": class_names,
        "metadata": metadata,
        "full_ds": full_ds,
        "engine": engine,
        "galleries": galleries,
        "gallery_ids": gallery_ids,
    }


P = load_pipeline()
CFG = P["cfg"]
MODS = list(CFG["modalities"])
BACKBONE = CFG["model"].get("backbone", "resnet18")


def _img_html(idx: int, modality: str, height: int = 96) -> str:
    rgb = render_patch(P["patches"], int(idx), modality)
    import io
    import base64

    buf = io.BytesIO()
    plt.imsave(buf, rgb, format="png")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" style="height:{height}px;border-radius:8px;border:1px solid #333"/>'


def _meta_str(rec: dict) -> str:
    bits = []
    if rec.get("sensor"):
        bits.append(rec["sensor"])
    if rec.get("land_cover"):
        bits.append(rec["land_cover"])
    if rec.get("acquisition_date"):
        bits.append(rec["acquisition_date"])
    if rec.get("latitude") is not None:
        bits.append(f"({rec['latitude']:.3f}, {rec['longitude']:.3f})")
    return " · ".join(bits)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs(
    ["Home", "Retrieval", "Results", "Map", "Analytics", "Embeddings",
     "Explainability", "Gallery", "History"]
)


# ---------------------------------------------------------------- Home ----
with tabs[0]:
    st.title("🛰️ Cross-Modal Satellite Image Retrieval")
    st.markdown(
        "Retrieve satellite images across **optical / multispectral / SAR** "
        "modalities from a shared embedding space + FAISS."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", f"{CFG['dataset'].get('name')} ({P['labels'].shape[0]:,})")
    c2.metric("Classes", len(P["class_names"]))
    c3.metric("Modalities", ", ".join(MODS))
    c4.metric("Embedding dim", CFG["model"].get("embedding_dim"))
    st.info(
        f"Backbone: `{BACKBONE}` · index `{CFG['retrieval']['index']['type']}`/"
        f"`{CFG['retrieval']['index']['metric']}` · "
        f"gallery {max(P['galleries'][m].size for m in MODS):,} vectors per modality"
    )

# ---------------------------------------------------------- Retrieval ----
with tabs[1]:
    st.subheader("Retrieval")
    mode = st.radio("Query source", ["Dataset id", "Upload image"], horizontal=True)
    query_modality = st.selectbox("Query modality", MODS)
    gallery_modality = st.selectbox("Gallery modality", MODS)
    k = st.slider("Top-K", 1, 10, 5)
    use_rerank = st.toggle("Re-rank (geo)", value=False)
    candidate_k = st.slider("FAISS candidate_k", k, 100, 50, disabled=not use_rerank)

    query_img = None
    if mode == "Dataset id":
        qid = st.number_input("Image id", 0, P["labels"].shape[0] - 1, 12, step=1)
        query_indices = [int(qid)]
    else:
        up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if up is None:
            st.stop()
        query_indices = None

    if st.button("Retrieve", type="primary"):
        reranker = None
        if use_rerank:
            reranker = build_reranker({"enabled": True, "method": "geo"})
        gallery = P["galleries"][gallery_modality]
        if query_indices is not None:
            res = P["engine"].retrieve(
                gallery, query_indices, query_modality, k=k,
                candidate_k=candidate_k if use_rerank else None, reranker=reranker,
            )
        else:
            import io
            import time
            import torch
            from PIL import Image

            from src.data.metadata import ImageMetadata
            from src.retrieval.engine import RetrievalResult

            data = up.read()
            img = Image.open(io.BytesIO(data)).convert("L" if query_modality == "sar" else "RGB")
            size = int(CFG["dataset"].get("image_size", 64))
            img = img.resize((size, size))
            if query_modality == "sar":
                arr = np.asarray(img, dtype=np.float32)[None] / 255.0
            else:
                arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            tf = P["full_ds"].transforms.get(query_modality)
            if tf is not None:
                arr = tf(np.ascontiguousarray(arr))
            x = torch.from_numpy(arr)[None]
            emb = P["engine"].model.embed(x.to(P["device"]), query_modality).detach().cpu().numpy()
            search_k = min(candidate_k if use_rerank else k, gallery.size)
            t0 = time.perf_counter()
            scores, ids = gallery.index.search(emb, search_k)
            ms = (time.perf_counter() - t0) * 1000.0
            rerank_ms = 0.0
            if use_rerank:
                q_meta = ImageMetadata()
                c_metas = [P["full_ds"].metadata_for(int(j)) for j in ids[0]] if P["full_ds"].metadata else None
                t_r = time.perf_counter()
                rs = reranker.score(emb[0], gallery.embeddings[ids[0]], q_meta, c_metas)
                rerank_ms = (time.perf_counter() - t_r) * 1000.0
                top = np.argsort(-np.asarray(rs))[:k]
                ids = ids[0][top]
                scores = scores[0][top]
            else:
                ids = ids[0][:k]
                scores = scores[0][:k]
            res = RetrievalResult(
                query_ids=np.array([-1]),
                gallery_ids=gallery.indices[ids][None],
                scores=scores[None],
                query_labels=np.array([-1]),
                gallery_labels=gallery.labels[ids][None],
                k=int(k),
                search_times_ms=np.array([ms]),
                query_modality=query_modality,
                gallery_modality=gallery_modality,
                rerank_times_ms=np.array([rerank_ms]),
            )
            st.session_state["result"] = res
            st.session_state["query_modality"] = query_modality
            st.session_state["rerank_used"] = use_rerank
            st.success("Retrieval complete")
            st.stop()
        st.session_state["result"] = res
        st.session_state["query_modality"] = query_modality
        st.session_state["rerank_used"] = use_rerank
        st.success("Retrieval complete")

# ------------------------------------------------------------ Results ----
with tabs[2]:
    st.subheader("Results")
    if "result" not in st.session_state:
        st.info("Run a retrieval first.")
    else:
        res = st.session_state["result"]
        qm = st.session_state["query_modality"]
        records = result_records(
            res, P["full_ds"], P["class_names"],
            query_metadata=None, gallery_metadata=P["full_ds"].metadata,
        )
        for rec in records:
            c1, c2 = st.columns([1, 3])
            qid = rec["query"]["image_id"]
            if qid >= 0:
                c1.markdown(_img_html(qid, qm, 140), unsafe_allow_html=True)
                c1.caption(f"QUERY #{qid} ({qm})")
            c2.markdown(
                f"search **{rec['search_time_ms']:.3f} ms** · "
                f"rerank **{rec['rerank_time_ms']:.3f} ms**"
            )
            for r in rec["retrieved"]:
                st.markdown(
                    f"**#{r['rank']}**  {_img_html(r['image_id'], r['modality'], 96)}"
                    f"  —  similarity `{r['similarity_score']}` · {_meta_str(r)}",
                    unsafe_allow_html=True,
                )

# ----------------------------------------------------------------- Map ----
with tabs[3]:
    st.subheader("Map")
    if "result" not in st.session_state or not P["full_ds"].metadata:
        st.info("Run a retrieval (with metadata available) to see locations.")
    else:
        res = st.session_state["result"]
        records = result_records(res, P["full_ds"], P["class_names"])
        rows = []
        for rec in records:
            qid = rec["query"]["image_id"]
            q_meta = P["full_ds"].metadata_for(int(qid)) if qid >= 0 else None
            if q_meta and q_meta.latitude is not None:
                rows.append({"lat": q_meta.latitude, "lon": q_meta.longitude, "type": "query", "label": f"Query #{qid}"})
            for r in rec["retrieved"]:
                meta = P["full_ds"].metadata_for(int(r["image_id"]))
                if meta and meta.latitude is not None:
                    rows.append({"lat": meta.latitude, "lon": meta.longitude,
                                 "type": "retrieved", "label": f"#{r['rank']} · {_meta_str(r)}"})
        if rows:
            df = pd.DataFrame(rows)
            import plotly.express as px

            fig = px.scatter_mapbox(df, lat="lat", lon="lon", color="type",
                                    hover_name="label", zoom=3, height=520)
            fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No coordinates available for the retrieved items.")

# ----------------------------------------------------------- Analytics ----
with tabs[4]:
    st.subheader("Analytics")
    metrics_path = os.path.join(ROOT, "outputs", "metrics", "retrieval_summary.json")
    if os.path.exists(metrics_path):
        import json

        with open(metrics_path, encoding="utf-8") as fh:
            data = json.load(fh)
        s = data.get("summary", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Same-modal F1@5", f"{s.get('same_modal_avg', {}).get('f1@5', 0):.3f}")
        c2.metric("Same-modal F1@10", f"{s.get('same_modal_avg', {}).get('f1@10', 0):.3f}")
        c3.metric("Cross-modal F1@5", f"{s.get('cross_modal_avg', {}).get('f1@5', 0):.3f}")
        c4.metric("Cross-modal F1@10", f"{s.get('cross_modal_avg', {}).get('f1@10', 0):.3f}")
        df = pd.DataFrame(data.get("rows", []))
        if not df.empty:
            st.dataframe(df)
        st.caption("Run `python run_pipeline.py` to (re)generate these metrics.")
    else:
        st.info("No evaluation report found. Run `python run_pipeline.py` first.")

# ---------------------------------------------------------- Embeddings ----
with tabs[5]:
    st.subheader("Embeddings")
    method = st.selectbox("Projection", ["pca", "tsne", "umap"])
    color_by = st.selectbox("Colour by", ["class", "modality"])
    n_sample = st.slider("Samples", 200, min(3000, P["labels"].shape[0]), 1000)
    if st.button("Project"):
        idx = np.random.RandomState(0).choice(len(P["full_ds"]), min(n_sample, len(P["full_ds"])), replace=False)
        mod = MODS[0]
        emb, _ = P["engine"].embed(idx, mod)
        if color_by == "class":
            fig = plot_embeddings(emb, P["labels"][idx], P["class_names"], method=method, title=f"By class ({method})")
        else:
            reps = np.concatenate(
                [P["engine"].embed(idx, m)[0] for m in MODS], axis=0)
            rep_mods = np.concatenate([np.full(len(idx), i) for i in range(len(MODS))])
            fig = plot_embeddings(reps, rep_mods, MODS, method=method, title=f"By modality ({method})")
        st.pyplot(fig)

# -------------------------------------------------------- Explainability ----
with tabs[6]:
    st.subheader("Explainability")
    qid = st.number_input("Query id", 0, P["labels"].shape[0] - 1, 12, step=1)
    modality = st.selectbox("Modality", MODS, key="xai_mod")
    if st.button("Explain", type="primary"):
        sample = P["full_ds"][int(qid)]
        import torch

        x = sample[modality][None].to(P["device"])
        rgb = render_patch(P["patches"], int(qid), modality)
        if BACKBONE in ("vit_b_16", "vit"):
            sal = attention_map(P["engine"].model, x, modality)
            title = "ViT attention"
        else:
            sal = gradcam(P["engine"].model, x, modality, class_idx=int(P["labels"][qid]))
            title = "Grad-CAM"
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb); axes[0].set_title(f"Query ({modality})")
        axes[1].imshow(sal, cmap="jet"); axes[1].set_title(title)
        axes[2].imshow(overlay_saliency(rgb, sal)); axes[2].set_title("Overlay")
        for ax in axes:
            ax.axis("off")
        st.pyplot(fig)

# ------------------------------------------------------------- Gallery ----
with tabs[7]:
    st.subheader("Gallery")
    mod = st.selectbox("Modality", MODS, key="gallery_mod")
    start = st.number_input("Start id", 0, P["labels"].shape[0] - 1, 0, step=12)
    n_show = 12
    cols = st.columns(4)
    for j, i in enumerate(range(int(start), min(int(start) + n_show, P["labels"].shape[0]))):
        with cols[j % 4]:
            st.markdown(_img_html(i, mod, 96), unsafe_allow_html=True)
            st.caption(f"#{i} {P['class_names'][int(P['labels'][i])]}")

# ------------------------------------------------------------- History ----
with tabs[8]:
    st.subheader("History")
    try:
        from src.database import MetadataStore

        store = MetadataStore(os.path.join(ROOT, "database", "metadata.db"))
        rows = store.recent_retrievals(20)
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df)
        else:
            st.info("No retrievals logged yet.")
    except Exception as exc:  # pragma: no cover
        st.warning(f"Could not read history: {exc}")


st.caption("Cross-Modal Satellite Image Retrieval · Streamlit UI · BHASHINI Problem 11")
