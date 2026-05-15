# Streamlit Demo — Handoff Document
### Visual Product Search Engine | DeepFashion In-Shop Clothes Retrieval

---

## What You're Building

A web app where a user uploads a clothing image, the system detects and crops the item, and returns the top-K visually + semantically similar products from the gallery.

---

## Files You Will Receive (physically handed to you)

| File | Size | What it is |
|------|------|------------|
| `hnsw_index_best.bin` | ~50–200 MB | Pre-built HNSW vector index (Config C, α=0.7, cosine space, 512-dim) |
| `gallery_metadata.csv` | ~2 MB | Gallery image paths + captions + item IDs |
| `ablation_results.csv` | tiny | Results table (for display if needed) |
| `best_clip.pt` | 837 MB | Fine-tuned CLIP ViT-B/32 checkpoint |
| `yolov8n (1).pt` | ~6 MB | Fine-tuned YOLOv8n clothing detector |

### Dataset
You already have the DeepFashion dataset. The gallery images are inside it at:
```
<your_dataset_root>/img/img/MEN/...
<your_dataset_root>/img/img/WOMEN/...
```

### gallery_metadata.csv schema
```
item_id, crop_path, caption, gender, category
```
`crop_path` currently points to Kaggle paths — **you need to update it** to match where your dataset lives locally. See the path fix below.

### ⚠️ Path Fix (important)
The `crop_path` column in `gallery_metadata.csv` has paths like:
```
/kaggle/input/datasets/svnsaisathvik/text-images/img/img/MEN/Denim/id_xxx/01_1.jpg
```
Replace the prefix with your local dataset root at the start of `app.py`:
```python
import pandas as pd
from pathlib import Path

LOCAL_IMG_ROOT = Path("path/to/your/dataset/img/img")  # ← change this
gallery_meta   = pd.read_csv("gallery_metadata.csv")

def fix_path(p):
    marker = "/img/img/"
    idx = str(p).find(marker)
    if idx != -1:
        return str(LOCAL_IMG_ROOT / str(p)[idx + len(marker):])
    return p

gallery_meta["crop_path"] = gallery_meta["crop_path"].apply(fix_path)
```

---

## What the App Must Do (per project rubric)

### Step-by-step flow:

**1. User uploads a query image**

**2. Run YOLO detection**
- Load `yolov8n (1).pt`
- Detect the clothing item
- Show the original image with bounding box drawn on it
- Crop to the detected region
- If no detection → use full image as crop

**3. Show crop + ask for confirmation**
- Display the cropped clothing region
- Two buttons: **"Confirm Crop"** / **"Use Full Image Instead"**
- Wait for user to confirm before proceeding

**4. After confirmation — encode query**
Use the fusion formula from the spec:
```
v_query = α · φ_V(crop) + (1-α) · φ_T(caption)
‖v_query‖ = 1
```
- `φ_V` = CLIP vision encoder (fine-tuned `best_clip.pt`)
- `φ_T` = CLIP text encoder (frozen)
- α is controlled by sidebar slider (default 0.7)
- For the query, you have no BLIP-2 caption → generate one using pretrained BLIP-2, OR just use α=1.0 for query (vision only) and α from slider for gallery. Either approach is acceptable.

**5. HNSW search**
- Load `hnsw_index_best.bin`
- Search for top-K results (K selectable: 5, 10, 15)
- Returns gallery row indices

**6. BLIP-2 ITM Re-ranking** *(required by project spec)*
- For each of the top-K candidates, compute BLIP-2 Image-Text Matching (ITM) score between the **query crop image** and the **candidate's caption** (from gallery_metadata.csv)
- Re-rank the top-K by ITM score
- Use `Salesforce/blip2-opt-2.7b` or `Salesforce/blip-image-captioning-base` for ITM (lighter option: `Salesforce/blip-itm-base-coco`)
- Recommended lighter model: `BlipForImageTextRetrieval` from `transformers`

```python
from transformers import BlipForImageTextRetrieval, BlipProcessor

itm_model     = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
itm_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

def blip_itm_score(image: PIL.Image, caption: str) -> float:
    inputs = itm_processor(images=image, text=caption,
                            return_tensors="pt").to(device)
    with torch.no_grad():
        score = itm_model(**inputs, use_itm_head=True).itm_score
    return torch.softmax(score, dim=1)[:, 1].item()  # P(match)
```

**7. Display results**
For each of the top-K re-ranked results show:
- Crop image
- Caption
- Cosine similarity score (from HNSW)
- ITM score (from BLIP-2)
- Item ID

---

## Sidebar Controls

| Control | Type | Default |
|---------|------|---------|
| K (number of results) | Selectbox: 5, 10, 15 | 10 |
| α (image vs text weight) | Slider: 0.0 → 1.0 | 0.7 |

---

## Tech Stack

```
streamlit
ultralytics          # YOLO
transformers         # CLIP + BLIP-2
hnswlib              # vector search
Pillow
torch
pandas
```

### requirements.txt
```
streamlit
ultralytics
transformers>=4.35.0
accelerate
hnswlib
Pillow
torch
pandas
```

---

## Loading Code Snippets

### Load CLIP (fine-tuned)
```python
from transformers import CLIPModel, CLIPProcessor
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

state = torch.load("best_clip.pt", map_location="cpu")
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]
model.load_state_dict(state, strict=False)
model.eval().to(device)
```

### Extract CLIP features (use this — avoids transformers version bugs)
```python
def get_image_embed(model, processor, image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.vision_model(**inputs)
        emb = model.visual_projection(out.pooler_output)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy()

def get_text_embed(model, processor, caption):
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
```

### Load HNSW index
```python
import hnswlib
import pandas as pd

gallery_meta = pd.read_csv("gallery_metadata.csv")

index = hnswlib.Index(space="cosine", dim=512)
index.load_index("hnsw_index_best.bin",
                  max_elements=len(gallery_meta))
index.set_ef(50)
```

### Search
```python
labels, distances = index.knn_query(query_embed, k=K)
top_k_rows = gallery_meta.iloc[labels[0]]
```

### Load YOLO
```python
from ultralytics import YOLO

yolo = YOLO("yolov8n (1).pt")

def detect_and_crop(image: PIL.Image) -> PIL.Image:
    results = yolo(image)
    if results[0].boxes:
        box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
        return image.crop((box[0], box[1], box[2], box[3]))
    return image   # fallback: full image
```

---

## Important Notes

1. **Use `@st.cache_resource`** for loading CLIP, BLIP-2, YOLO, and HNSW index — otherwise they reload on every interaction and the app becomes unusably slow.

2. **gallery_metadata `crop_path`** has Kaggle-style paths — apply the path fix at the top of `app.py` (see above) to point to your local dataset folder.

3. **BLIP-2 ITM is required** by the project spec under the online query pipeline. Use `blip-itm-base-coco` (lightweight, ~900MB) not the full BLIP-2 (3GB+).

4. **Do NOT use `st.form`** — use regular `st.button` for interactions.

5. The HNSW index was built with `Config C, α=0.7, fine-tuned CLIP`. Gallery embeddings are already stored in it. You only need to encode the **query** at runtime.

---

## Expected App Flow (for viva demo)

```
Upload image
    → YOLO crops clothing
    → Show crop, user confirms
    → CLIP encodes query crop
    → HNSW returns top-K
    → BLIP-2 ITM re-ranks top-K
    → Display results with scores
```

---

## How to Run Locally

```bash
pip install streamlit ultralytics transformers accelerate hnswlib Pillow torch pandas
streamlit run app.py
```
Opens at `http://localhost:8501`

Put all received files in the same folder as `app.py`:
```
your_folder/
├── app.py
├── hnsw_index_best.bin
├── gallery_metadata.csv
├── ablation_results.csv
├── best_clip.pt
└── yolov8n (1).pt
```

---

*Handoff prepared: May 14, 2026*
*Best model: Config C, seed 525, val_loss=0.5950*
