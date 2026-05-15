"""
eval_batch.py — Batch Evaluation Script
Visual Product Search Engine | DeepFashion In-Shop Clothes Retrieval

Usage:
    python eval_batch.py \
        --query_dir     /path/to/query_images/ \
        --gallery_csv   /path/to/gallery_metadata.csv \
        --index_path    /path/to/hnsw_index_best.bin \
        --clip_ckpt     /path/to/best_clip.pt \
        --yolo_ckpt     /path/to/yolov8n.pt \
        --alpha         0.7 \
        --k             5 10 15

Outputs:
    - Prints Recall@K, NDCG@K, mAP@K for each K
    - Saves results to eval_results.csv
"""

import argparse
import numpy as np
import pandas as pd
import torch
import hnswlib
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ── Optional YOLO import ───────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠  ultralytics not installed — YOLO crop disabled, using full image")

# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Batch Evaluation Script")
    parser.add_argument("--query_dir",   required=True,
                        help="Folder containing query images (jpg/png)")
    parser.add_argument("--gallery_csv", required=True,
                        help="Path to gallery_metadata.csv")
    parser.add_argument("--index_path",  required=True,
                        help="Path to hnsw_index_best.bin")
    parser.add_argument("--clip_ckpt",   required=True,
                        help="Path to best_clip.pt")
    parser.add_argument("--yolo_ckpt",   default=None,
                        help="Path to yolov8n.pt (optional)")
    parser.add_argument("--alpha",       type=float, default=0.7,
                        help="Fusion weight α (default: 0.7)")
    parser.add_argument("--k",           type=int, nargs="+", default=[5, 10, 15],
                        help="K values for metrics (default: 5 10 15)")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--output_csv",  default="eval_results.csv",
                        help="Where to save results table")
    parser.add_argument("--img_root",    default=None,
                        help="Local dataset root to remap gallery crop paths "
                             "(e.g. /path/to/dataset/img/img)")
    return parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_clip(ckpt_path):
    print(f"Loading CLIP from {ckpt_path} …")
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    state     = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval().to(DEVICE)
    print("  CLIP loaded ✅")
    return model, processor


def get_image_embed(model, processor, image: Image.Image) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.vision_model(**inputs)
        emb = model.visual_projection(out.pooler_output)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy()


def get_text_embed(model, processor, caption: str) -> np.ndarray:
    inputs = processor(text=caption, return_tensors="pt",
                       truncation=True, max_length=77).to(DEVICE)
    with torch.no_grad():
        out = model.text_model(**inputs)
        emb = model.text_projection(out.pooler_output)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy()


def fuse(img_emb: np.ndarray, txt_emb: np.ndarray, alpha: float) -> np.ndarray:
    """v = α·φ_V + (1-α)·φ_T,  ‖v‖=1"""
    v = alpha * img_emb + (1.0 - alpha) * txt_emb
    return (v / np.linalg.norm(v)).astype(np.float32)


def yolo_crop(yolo_model, image: Image.Image) -> Image.Image:
    """Run YOLO detection and return cropped clothing region."""
    results = yolo_model(image, verbose=False)
    if results and results[0].boxes and len(results[0].boxes):
        box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        cropped = image.crop((x1, y1, x2, y2))
        if cropped.size[0] > 0 and cropped.size[1] > 0:
            return cropped
    return image   # fallback: full image


def fix_gallery_paths(gallery_df: pd.DataFrame, img_root: str) -> pd.DataFrame:
    """Remap Kaggle-style crop_path to local dataset root."""
    root    = Path(img_root)
    marker  = "/img/img/"

    def remap(p):
        idx = str(p).find(marker)
        if idx != -1:
            rel = str(p)[idx + len(marker):]
            return str(root / rel)
        return p

    gallery_df = gallery_df.copy()
    gallery_df["crop_path"] = gallery_df["crop_path"].apply(remap)
    return gallery_df

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(retrieved, query_ids, gallery_ids, k):
    hits = sum(
        1 for i, qid in enumerate(query_ids)
        if qid in gallery_ids[retrieved[i, :k]]
    )
    return hits / len(query_ids)


def ndcg_at_k(retrieved, query_ids, gallery_ids, k):
    scores = []
    for i, qid in enumerate(query_ids):
        top    = gallery_ids[retrieved[i, :k]]
        dcg    = sum(1.0 / np.log2(r + 2) for r, g in enumerate(top) if g == qid)
        n_rel  = int(np.sum(gallery_ids == qid))
        ideal  = min(n_rel, k)
        idcg   = sum(1.0 / np.log2(r + 2) for r in range(ideal))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores))


def map_at_k(retrieved, query_ids, gallery_ids, k):
    ap_list = []
    for i, qid in enumerate(query_ids):
        top      = gallery_ids[retrieved[i, :k]]
        hits, ap = 0, 0.0
        for rank, gid in enumerate(top, 1):
            if gid == qid:
                hits += 1
                ap   += hits / rank
        n_rel = int(np.sum(gallery_ids == qid))
        denom = min(n_rel, k) if n_rel > 0 else 1
        ap_list.append(ap / denom)
    return float(np.mean(ap_list))

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load gallery metadata ─────────────────────────────────────────────────
    print(f"\nLoading gallery metadata from {args.gallery_csv} …")
    gallery_df = pd.read_csv(args.gallery_csv)
    if args.img_root:
        gallery_df = fix_gallery_paths(gallery_df, args.img_root)
    print(f"  Gallery size: {len(gallery_df):,} images")

    # ── Load HNSW index ───────────────────────────────────────────────────────
    print(f"\nLoading HNSW index from {args.index_path} …")
    index = hnswlib.Index(space="cosine", dim=512)
    index.load_index(args.index_path, max_elements=len(gallery_df))
    index.set_ef(50)
    print("  HNSW index loaded ✅")

    # ── Load CLIP ─────────────────────────────────────────────────────────────
    model, processor = load_clip(args.clip_ckpt)

    # ── Load YOLO (optional) ──────────────────────────────────────────────────
    yolo_model = None
    if args.yolo_ckpt and YOLO_AVAILABLE:
        print(f"\nLoading YOLO from {args.yolo_ckpt} …")
        yolo_model = YOLO(args.yolo_ckpt)
        print("  YOLO loaded ✅")

    # ── Collect query images ──────────────────────────────────────────────────
    query_dir   = Path(args.query_dir)
    image_files = sorted(
        list(query_dir.glob("**/*.jpg")) +
        list(query_dir.glob("**/*.jpeg")) +
        list(query_dir.glob("**/*.png"))
    )
    print(f"\nFound {len(image_files):,} query images in {query_dir}")

    if len(image_files) == 0:
        print("❌ No images found. Check --query_dir path.")
        return

    # ── Derive item_id from filename ──────────────────────────────────────────
    # Expected filename format: id_00000080_01_1_front.jpg
    # or folder structure:      id_00000080/01_1_front.jpg
    def extract_item_id(path: Path) -> str:
        # Try parent folder name first
        if path.parent.name.startswith("id_"):
            return path.parent.name
        # Try filename prefix
        name = path.stem
        if name.startswith("id_"):
            return "_".join(name.split("_")[:3])   # id_00000080
        return path.stem   # fallback

    # ── Encode query images ───────────────────────────────────────────────────
    print(f"\nEncoding {len(image_files):,} query images (α={args.alpha}) …")
    query_embeds = []
    query_ids    = []
    failed       = 0

    for img_path in tqdm(image_files, desc="Encoding queries"):
        try:
            image = Image.open(img_path).convert("RGB")

            # YOLO crop
            if yolo_model is not None:
                image = yolo_crop(yolo_model, image)

            # Image embed
            img_emb = get_image_embed(model, processor, image)

            # Fuse with text if alpha < 1
            if args.alpha < 1.0:
                # Use a generic caption if no caption available for query
                caption = "a clothing item"
                txt_emb = get_text_embed(model, processor, caption)
                emb     = fuse(img_emb, txt_emb, args.alpha)
            else:
                emb = img_emb.astype(np.float32)

            query_embeds.append(emb.flatten())
            query_ids.append(extract_item_id(img_path))

        except Exception as e:
            print(f"  ⚠  Failed on {img_path.name}: {e}")
            failed += 1

    if failed > 0:
        print(f"  {failed} images failed to encode and were skipped")

    query_embeds = np.vstack(query_embeds).astype(np.float32)
    query_ids    = np.array(query_ids)
    gallery_ids  = gallery_df["item_id"].values

    print(f"\nQuery embeddings shape  : {query_embeds.shape}")
    print(f"Gallery embeddings count: {len(gallery_df):,}")

    # ── HNSW search ───────────────────────────────────────────────────────────
    print("\nSearching HNSW index …")
    max_k          = max(args.k)
    labels, _      = index.knn_query(query_embeds, k=max_k)

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"  EVALUATION RESULTS  (α={args.alpha})")
    print("=" * 50)

    rows = []
    for k in sorted(args.k):
        r = recall_at_k(labels, query_ids, gallery_ids, k)
        n = ndcg_at_k(  labels, query_ids, gallery_ids, k)
        m = map_at_k(   labels, query_ids, gallery_ids, k)
        print(f"  Recall@{k:<3}: {r:.4f}  |  NDCG@{k:<3}: {n:.4f}  |  mAP@{k:<3}: {m:.4f}")
        rows.append({"K": k, "Recall": r, "NDCG": n, "mAP": m})

    print("=" * 50)

    # ── Save results ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(rows)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved → {args.output_csv}")


if __name__ == "__main__":
    main()
