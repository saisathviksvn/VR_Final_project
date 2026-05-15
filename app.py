import streamlit as st
import numpy as np
import pandas as pd
import torch
import hnswlib
import PIL.Image
import PIL.ImageDraw
from pathlib import Path
from transformers import (
    CLIPModel, CLIPProcessor,
    BlipForImageTextRetrieval, BlipProcessor,
    AutoImageProcessor, AutoModelForObjectDetection,
)

# ─────────────────────────────────────────────
# PATHS  — adjust if you move things
# ─────────────────────────────────────────────
EMOBRO          = Path("/Users/saiganesh/Documents/emobro")
DATASET_ROOT    = Path("/Users/saiganesh/Documents/vr_project_dataset")
LOCAL_IMG_ROOT  = DATASET_ROOT / "img" / "img"

CLIP_CKPT       = DATASET_ROOT / "best_clip.pt"
HNSW_INDEX      = EMOBRO / "hnsw_index_best.bin"
GALLERY_CSV     = EMOBRO / "gallery_metadata.csv"
ABLATION_CSV    = EMOBRO / "ablation_results.csv"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FashionSearch",
    page_icon="🧥",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0e0e0e;
    color: #f0ede6;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
    color: #f0ede6;
}
.stButton > button {
    background: #f0ede6;
    color: #0e0e0e;
    border: none;
    border-radius: 2px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    letter-spacing: 0.05em;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #c8b8a2;
    color: #0e0e0e;
}
.result-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
}
.score-badge {
    display: inline-block;
    background: #2a2a2a;
    color: #c8b8a2;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 2px;
    margin-right: 4px;
    font-family: 'DM Sans', monospace;
    letter-spacing: 0.03em;
}
.item-id {
    font-size: 0.65rem;
    color: #555;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.caption-text {
    font-size: 0.82rem;
    color: #aaa;
    margin: 0.3rem 0;
    font-style: italic;
}
.section-label {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CLIP model…")
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    state = torch.load(str(CLIP_CKPT), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model, processor, device

# Fashionpedia YOLOS — categories we actually care about for retrieval
CLOTHING_ALLOWLIST = {
    "shirt", "t-shirt", "top", "blouse", "tank top", "polo shirt",
    "jacket", "coat", "blazer", "hoodie", "sweatshirt", "sweater",
    "cardigan", "vest", "windbreaker", "parka", "trench coat",
    "jeans", "pants", "trousers", "chinos", "leggings", "joggers",
    "skirt", "shorts", "dress", "jumpsuit", "romper", "overalls",
    "saree", "kurta", "suit", "tuxedo", "robe", "kimono",
}
CONF_THRESHOLD  = 0.30   # minimum confidence to show a detection
MIN_BOX_AREA    = 2000   # ignore tiny noise boxes (px²)

@st.cache_resource(show_spinner="Loading Fashionpedia detector…")
def load_fashionpedia():
    processor = AutoImageProcessor.from_pretrained("valentinafevu/yolos-fashionpedia")
    model     = AutoModelForObjectDetection.from_pretrained("valentinafevu/yolos-fashionpedia")
    model.eval()
    return processor, model

def run_fashionpedia(image: PIL.Image.Image, processor, model):
    """
    Returns list of dicts sorted by confidence desc:
      { label, confidence, box (x0,y0,x1,y1), crop }
    Only valid clothing items above threshold with reasonable area.
    """
    inputs  = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])   # (H, W)
    results = processor.post_process_object_detection(
        outputs, threshold=CONF_THRESHOLD, target_sizes=target_sizes
    )[0]

    detections = []
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        label = model.config.id2label[label_id.item()].lower().strip()
        conf  = score.item()
        x0, y0, x1, y1 = [int(v) for v in box.tolist()]
        # clamp to image bounds
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(image.width, x1), min(image.height, y1)
        area = (x1 - x0) * (y1 - y0)
        if area < MIN_BOX_AREA:
            continue
        # check against allowlist (substring match for compound labels)
        if not any(kw in label for kw in CLOTHING_ALLOWLIST):
            continue
        crop = image.crop((x0, y0, x1, y1))
        detections.append({
            "label":      label,
            "confidence": conf,
            "box":        (x0, y0, x1, y1),
            "crop":       crop,
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections

def draw_all_boxes(image: PIL.Image.Image, detections):
    """Draw labeled boxes for all detections on a copy of the image."""
    annotated = image.copy()
    draw = PIL.ImageDraw.Draw(annotated)
    colors = ["#c8b8a2", "#a2c8b8", "#b8a2c8", "#c8a2a2", "#a2b8c8"]
    for i, det in enumerate(detections):
        color = colors[i % len(colors)]
        x0, y0, x1, y1 = det["box"]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label_text = f"{det['label']} {det['confidence']:.0%}"
        draw.rectangle([x0, y0 - 18, x0 + len(label_text) * 7, y0], fill=color)
        draw.text((x0 + 2, y0 - 16), label_text, fill="#0e0e0e")
    return annotated

@st.cache_resource(show_spinner="Loading BLIP-2 ITM…")
def load_blip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    model.eval().to(device)
    return model, processor, device

@st.cache_resource(show_spinner="Loading HNSW index…")
def load_index_and_meta():
    meta = pd.read_csv(str(GALLERY_CSV))

    def fix_path(p):
        marker = "/img/img/"
        idx = str(p).find(marker)
        if idx != -1:
            return str(LOCAL_IMG_ROOT / str(p)[idx + len(marker):])
        return p

    meta["crop_path"] = meta["crop_path"].apply(fix_path)

    index = hnswlib.Index(space="cosine", dim=512)
    index.load_index(str(HNSW_INDEX), max_elements=len(meta))
    index.set_ef(50)
    return index, meta

# ─────────────────────────────────────────────
# EMBEDDING HELPERS
# ─────────────────────────────────────────────
def get_image_embed(model, processor, image, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.vision_model(**inputs)
        emb = model.visual_projection(out.pooler_output)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy()

def get_text_embed(model, processor, caption, device):
    inputs = processor(text=caption, return_tensors="pt",
                       truncation=True, max_length=77).to(device)
    with torch.no_grad():
        out = model.text_model(**inputs)
        emb = model.text_projection(out.pooler_output)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy()

def fuse(img_emb, txt_emb, alpha):
    v = alpha * img_emb + (1 - alpha) * txt_emb
    return (v / np.linalg.norm(v)).astype(np.float32)

def blip_itm_score(image, caption, blip_model, blip_proc, device):
    inputs = blip_proc(images=image, text=caption,
                       return_tensors="pt").to(device)
    with torch.no_grad():
        score = blip_model(**inputs, use_itm_head=True).itm_score
    return torch.softmax(score, dim=1)[:, 1].item()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    K = st.selectbox("Results (K)", [5, 10, 15], index=1)
    alpha = st.slider("α — vision weight", 0.0, 1.0, 0.7, 0.05,
                      help="1.0 = vision only, 0.0 = text only")
    st.markdown("---")
    st.markdown("**Model:** Config C · α=0.7")
    st.markdown("**Index:** HNSW cosine · 512-dim")
    st.markdown("**val_loss:** 0.5950")

    if ABLATION_CSV.exists():
        with st.expander("Ablation results"):
            st.dataframe(pd.read_csv(str(ABLATION_CSV)), use_container_width=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 🧥 FashionSearch")
st.markdown('<p class="section-label">Visual Product Retrieval · DeepFashion In-Shop</p>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# STEP 1 — UPLOAD
# ─────────────────────────────────────────────
uploaded = st.file_uploader("Upload a clothing image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.info("Upload a clothing image to begin.")
    st.stop()

query_img = PIL.Image.open(uploaded).convert("RGB")

# ─────────────────────────────────────────────
# STEP 2 — FASHIONPEDIA MULTI-ITEM DETECTION
# ─────────────────────────────────────────────
fp_processor, fp_model = load_fashionpedia()

col1, col2 = st.columns(2)
with col1:
    st.markdown('<p class="section-label">Uploaded image</p>', unsafe_allow_html=True)
    st.image(query_img, use_container_width=True)

with st.spinner("Detecting clothing items…"):
    detections = run_fashionpedia(query_img, fp_processor, fp_model)

with col2:
    if detections:
        st.markdown('<p class="section-label">All detected clothing items</p>', unsafe_allow_html=True)
        st.image(draw_all_boxes(query_img, detections), use_container_width=True)
    else:
        st.warning("No clothing items detected above threshold.")

# ─────────────────────────────────────────────
# STEP 3 — USER SELECTS WHICH ITEM TO SEARCH
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-label">Select a clothing item to search</p>', unsafe_allow_html=True)

if not detections:
    # Fallback: let user proceed with full image
    st.info("No detections — using the full image as query.")
    if st.button("🔍 Search with Full Image"):
        st.session_state["final_crop"]      = query_img
        st.session_state["crop_confirmed"]  = True
else:
    # Show all detections as a grid with "Search this" buttons
    cols_det = st.columns(min(len(detections), 5))
    for i, det in enumerate(detections):
        with cols_det[i % 5]:
            st.image(det["crop"], use_container_width=True)
            st.markdown(
                f'<p class="caption-text">{det["label"]}</p>'
                f'<span class="score-badge">conf {det["confidence"]:.0%}</span>',
                unsafe_allow_html=True
            )
            if st.button(f"🔍 Search this", key=f"sel_{i}"):
                st.session_state["final_crop"]     = det["crop"]
                st.session_state["crop_confirmed"] = True
                st.session_state["selected_label"] = det["label"]

if not st.session_state.get("crop_confirmed"):
    st.stop()

final_crop = st.session_state["final_crop"]
selected_label = st.session_state.get("selected_label", "full image")

st.success(f"Searching using: **{selected_label}**")


# ─────────────────────────────────────────────
# STEP 4 — ENCODE QUERY
# ─────────────────────────────────────────────
clip_model, clip_proc, clip_device = load_clip()

with st.spinner("Encoding query…"):
    # Vision-only for query (no caption available at query time)
    img_emb   = get_image_embed(clip_model, clip_proc, final_crop, clip_device)
    # Use alpha=1.0 for query (vision only) as per handoff note
    query_vec = img_emb.astype(np.float32)

# ─────────────────────────────────────────────
# STEP 5 — HNSW SEARCH
# ─────────────────────────────────────────────
index, gallery_meta = load_index_and_meta()

with st.spinner(f"Searching index for top-{K}…"):
    labels, distances = index.knn_query(query_vec, k=K)
    top_rows = gallery_meta.iloc[labels[0]].reset_index(drop=True)
    cosine_sims = 1 - distances[0]   # hnswlib returns distances not similarities

# ─────────────────────────────────────────────
# STEP 6 — BLIP-2 ITM RE-RANKING
# ─────────────────────────────────────────────
blip_model, blip_proc, blip_device = load_blip()

itm_scores = []
progress = st.progress(0, text="Computing ITM scores…")
for i, (_, row) in enumerate(top_rows.iterrows()):
    caption = str(row["caption"])
    score   = blip_itm_score(final_crop, caption, blip_model, blip_proc, blip_device)
    itm_scores.append(score)
    progress.progress((i + 1) / len(top_rows), text=f"ITM {i+1}/{len(top_rows)}")

progress.empty()

top_rows["cosine_sim"] = cosine_sims
top_rows["itm_score"]  = itm_scores
top_rows = top_rows.sort_values("itm_score", ascending=False).reset_index(drop=True)

# ─────────────────────────────────────────────
# STEP 7 — DISPLAY RESULTS
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(f"## Top {K} Results")
st.markdown('<p class="section-label">Re-ranked by BLIP-2 ITM score</p>', unsafe_allow_html=True)

cols_per_row = 5
for row_start in range(0, len(top_rows), cols_per_row):
    cols = st.columns(cols_per_row)
    for col_idx, col in enumerate(cols):
        idx = row_start + col_idx
        if idx >= len(top_rows):
            break
        row = top_rows.iloc[idx]
        with col:
            img_path = Path(row["crop_path"])
            if img_path.exists():
                result_img = PIL.Image.open(img_path).convert("RGB")
                st.image(result_img, use_container_width=True)
            else:
                st.markdown("*(image not found)*")

            st.markdown(
                f'<p class="item-id">#{idx+1} · {row["item_id"]}</p>'
                f'<p class="caption-text">{row["caption"]}</p>'
                f'<span class="score-badge">cos {row["cosine_sim"]:.3f}</span>'
                f'<span class="score-badge">itm {row["itm_score"]:.3f}</span>'
                f'<p class="item-id">{row.get("gender","")}&nbsp;·&nbsp;{row.get("category","")}</p>',
                unsafe_allow_html=True
            )

# Reset confirmation so user can re-upload cleanly
if st.button("🔄 Start Over"):
    for key in ["final_crop", "crop_confirmed", "selected_label"]:
        st.session_state.pop(key, None)
    st.rerun()
