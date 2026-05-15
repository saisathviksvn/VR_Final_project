# Visual Product Search Engine
### Query-by-Image Retrieval · DeepFashion In-Shop Clothes Retrieval

> **Visual Recognition Course — Final Project**
> International Institute of Information Technology, Bangalore · 2024–25

---

## 🧠 What This Does

Upload a photo of any clothing item → get back visually and semantically similar products from the catalog. No text search, no keyword matching — just the image.

The system combines:
- **YOLOv8n** — crops the clothing item out of the image
- **BLIP-2** — generates a natural language caption for each gallery item
- **CLIP (fine-tuned)** — encodes image + caption into a shared 512-d embedding
- **HNSW** — approximate nearest-neighbour index for fast retrieval
- **BLIP-2 ITM** — re-ranks candidates by image-text matching score at query time

---

## 📊 Results

Best configuration: **Fine-tuned CLIP, α = 0.7, Seed 525**

| Metric | @5 | @10 | @15 |
|---|---|---|---|
| Recall | 0.6910 | 0.7685 | **0.8033** |
| NDCG | 0.3729 | 0.3824 | **0.3927** |
| mAP | 0.2923 | 0.2863 | **0.2883** |

Multi-seed mean (seeds 1, 52, 525): **Recall@15 = 0.7956 ± 0.0075**

<details>
<summary>Full ablation table</summary>

| Config | α | R@5 | R@10 | R@15 | N@5 | N@10 | N@15 | mAP@5 | mAP@10 | mAP@15 |
|---|---|---|---|---|---|---|---|---|---|---|
| A: Frozen CLIP | 1.0 | 0.6721 | 0.7423 | 0.7804 | 0.3557 | 0.3617 | 0.3719 | 0.2748 | 0.2651 | 0.2666 |
| B: Frozen CLIP | 0.7 | 0.6891 | 0.7607 | 0.7974 | 0.3751 | 0.3846 | 0.3955 | 0.2951 | 0.2890 | 0.2911 |
| B: Frozen CLIP | 0.5 | 0.5506 | 0.6256 | 0.6726 | 0.2839 | 0.2931 | 0.3035 | 0.2193 | 0.2151 | 0.2171 |
| **C: Fine-tuned** | **0.7** | **0.6910** | **0.7685** | **0.8033** | **0.3729** | **0.3824** | **0.3927** | **0.2923** | **0.2863** | **0.2883** |
| C: Fine-tuned | 0.5 | 0.5720 | 0.6537 | 0.6993 | 0.2947 | 0.3049 | 0.3151 | 0.2270 | 0.2232 | 0.2251 |

</details>

---

## 🗂️ Repository Structure

```
VR_Final_project/
│
├── Crop_and_blip.ipynb                  # Notebook 1: YOLOv8n fine-tuning + BLIP-2 captioning
├── clip-finetuning_with_seeds.ipynb     # Notebook 2: CLIP fine-tuning (seeds 1, 52, 525)
├── vr-final-evaluation_metrics.ipynb   # Notebook 3: HNSW indexing + ablation + metrics
│
├── metadata_parthiv.csv                 # Gallery metadata: captions, item IDs, paths (14.2 MB)
│
└── streamlit_demoapp/
    ├── app.py                           # Interactive Streamlit demo
    ├── eval_batch.py                    # Standalone batch evaluation script
    ├── Streamlit_Handoff.md             # Architecture + setup docs
    ├── ablation_results.csv             # Precomputed ablation metrics
    ├── gallery_metadata.csv             # Gallery metadata for runtime annotation (1.9 MB)
    └── yolov8n.pt                       # Fine-tuned YOLOv8n weights (6.5 MB)
```

---

## ⚙️ Setup

### Prerequisites

```bash
pip install torch torchvision
pip install transformers
pip install ultralytics
pip install hnswlib
pip install streamlit
pip install pandas pillow tqdm
```

### Required Files (not in repo — too large)

You need to download and place these manually:

| File | What it is | Where to put it |
|---|---|---|
| `best_clip.pt` | Fine-tuned CLIP checkpoint (seed 525) | `streamlit_demoapp/` |
| `hnsw_index_best.bin` | Pre-built HNSW index (Config C, α=0.7) | `streamlit_demoapp/` |
| DeepFashion dataset | Raw images | anywhere, pass path via `--img_root` |

---

## 🚀 Running the Demo

```bash
cd streamlit_demoapp
streamlit run app.py
```

The app will open at `http://localhost:8501`. Use the sidebar to adjust α and top-K.

**Workflow:**
1. Upload a clothing image
2. YOLOv8n detects all clothing items — confirm the one you want
3. System encodes the crop using fused CLIP embeddings
4. HNSW returns top-K candidates, re-ranked by BLIP-2 ITM
5. Results shown with similarity scores, ITM scores, captions, and item IDs

---

## 📏 Batch Evaluation

Run the full retrieval pipeline over a folder of query images and compute all metrics:

```bash
python eval_batch.py \
    --query_dir   /path/to/query_images/ \
    --gallery_csv /path/to/gallery_metadata.csv \
    --index_path  /path/to/hnsw_index_best.bin \
    --clip_ckpt   /path/to/best_clip.pt \
    --yolo_ckpt   /path/to/yolov8n.pt \
    --alpha       0.7 \
    --k           5 10 15
```

Results are printed to stdout and saved to `eval_results.csv`.

**All arguments:**

| Argument | Default | Description |
|---|---|---|
| `--query_dir` | required | Folder of query images (jpg/png), searched recursively |
| `--gallery_csv` | required | Gallery metadata CSV with item IDs and crop paths |
| `--index_path` | required | Pre-built HNSW index (`.bin`) |
| `--clip_ckpt` | required | Fine-tuned CLIP checkpoint |
| `--yolo_ckpt` | None | YOLOv8n weights (optional; skips crop if absent) |
| `--alpha` | 0.7 | Image-text fusion weight |
| `--k` | 5 10 15 | K values for metric computation |
| `--img_root` | None | Remaps Kaggle-style crop paths to local dataset root |
| `--output_csv` | `eval_results.csv` | Output file for results |

---

## 🏗️ System Architecture

```
OFFLINE (runs once)
─────────────────────────────────────────────────────────
Raw Image → YOLOv8n Crop → BLIP-2 Caption → CLIP Fused Embed → HNSW Index

ONLINE (per query)
─────────────────────────────────────────────────────────
Query Image → YOLOv8n Crop → User Confirms → CLIP Embed
           → HNSW Search → BLIP-2 ITM Re-rank → Top-K Results
```

**Fusion formula:**

```
v_i = α · φ_V(crop_i) + (1 − α) · φ_T(caption_i),   ‖v_i‖ = 1
```

- `φ_V` = CLIP vision encoder, `φ_T` = CLIP text encoder
- α = 0.7 gives best results (more visual weight, less caption weight)

---

## 🔧 CLIP Fine-tuning Details

| Setting | Value |
|---|---|
| Base model | `openai/clip-vit-base-patch32` |
| Fine-tuned layers | Last 4 ViT blocks (layers 8–11) + projection |
| Frozen | Text encoder + first 8 ViT blocks |
| Trainable params | 29.01 M / 151.3 M total |
| Loss | InfoNCE contrastive loss |
| Batch sampler | Item-ID-aware (same item guaranteed in same batch) |
| Optimizer | AdamW + OneCycleLR |
| Epochs | 10 per seed |
| Seeds | 1, 52, 525 (team roll numbers) |
| Best checkpoint | Seed 525 (val loss = 0.5950) |

---

## 📦 Dataset

**DeepFashion In-Shop Clothes Retrieval** — Liu et al., CVPR 2016

| Split | Images | Role |
|---|---|---|
| Train | 25,882 | CLIP fine-tuning only |
| Query | 14,218 | Retrieval inputs |
| Gallery | 12,612 | Search database |
| **Total** | **52,712** | |

- 7,982 unique item IDs · MEN + WOMEN · 10+ categories
- Ground truth: two images match ↔ same `item_id`
- YOLO detection success rate: **97.4%** (2.6% centre-crop fallback)

---

## 👥 Team

| Name | Roll No. | Seed |
|---|---|---|
| SVN Sai Sathvik | IMT2023001 | 1 |
| K Kapil Aditya Reddy | IMT2023052 | 52 |
| Sai Ganesh Upadrasta | IMT2023525 | 525 |
| Kotyada Parthiv | IMT2023559 | — |

---

## 📄 References

- Radford et al. (2021) — CLIP, ICML
- Li et al. (2023) — BLIP-2, ICML
- Jocher et al. (2023) — YOLOv8, [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- Malkov & Yashunin (2018) — HNSW, IEEE TPAMI
- Liu et al. (2016) — DeepFashion, CVPR
