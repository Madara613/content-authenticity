# Multimodal Content Authenticity System

A multimodal verification platform that analyzes an image together with its caption/claim from news or social media, returning an honest authenticity verdict, confidence score, and top contributing signals.

---

## 🎯 Project Goal

When given an image and an accompanying claim/caption, the system evaluates both visual and textual evidence alongside cross-modal consistency to classify the submission into one of four labels:

1. **`Authentic`**: Verifiably genuine image and corroborating claim without tampering or synthetic artifacts.
2. **`Manipulated`**: Evidence of localized digital editing (splicing, copy-move) or contextual mismatch (cheapfakes / image reuse).
3. **`Synthetic`**: Statistical artifacts characteristic of generative AI (diffusion models, GANs).
4. **`Unverifiable`**: When signals conflict, evidence is insufficient, or confidence is low. Honest uncertainty is a first-class feature.

Every analysis is logged in Supabase Postgres for full reproducibility and auditing.

---

## 🔬 Modalities & Detection Signals

- **Image Forensics**:
  - EXIF & C2PA metadata parsing
  - Error Level Analysis (ELA) for compression anomalies
  - OpenCV noise variance analysis and copy-move keypoint clustering
  - Pretrained Hugging Face AI image detector
- **Text Analysis**:
  - Atomic claim extraction using Gemini API (`google-genai` SDK)
  - Evidence retrieval via Gemini Google Search grounding
  - Historical misinformation matching against local LIAR benchmark (Sentence-Transformers + FAISS)
  - Natural Language Inference (NLI) stance classification (Entailment / Neutral / Contradiction)
- **Cross-Modal Consistency**:
  - CLIP image-caption semantic similarity (`clip-vit-base-patch32`)
  - Perceptual hashing (`imagehash` dHash/pHash) for image reuse detection
- **Fusion Engine**:
  - Weighted evidence synthesis with explicit confidence gating and conflict resolution rules

---

## 📁 Repository Structure

```text
content-authenticity/
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore rules (secrets, data, caches)
├── requirements.txt        # Backend Python dependencies
├── README.md               # Project overview and quickstart
├── SPEC.md                 # Detailed architectural and API specification
├── backend/                # FastAPI backend service (Python 3.11)
├── frontend/               # React + Tailwind + Framer Motion (Vite)
├── notebooks/              # Jupyter evaluation and benchmark notebooks
├── data/                   # Local datasets (LIAR benchmark, FAISS indices)
└── docs/                   # Additional documentation & diagrams
```

---

## 🛠️ Tech Stack (Free Tools Only)

- **Backend**: Python 3.11, FastAPI, Uvicorn, Pydantic
- **Frontend**: React, Tailwind CSS, Framer Motion, Vite
- **Database**: Supabase Postgres (Free Tier)
- **AI & ML**: Google Gemini API (`google-genai` free tier), Hugging Face Transformers, PyTorch, OpenCV, Pillow, FAISS, Sentence-Transformers

---

## 🚀 Quickstart & Setup

### 1. Clone & Configure Environment

```bash
cp .env.example .env
# Open .env and insert your free Gemini API key and Supabase credentials
```

### 2. Backend Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 🗺️ Build Phases

- [ ] **Phase 1**: Backend skeleton + Metadata + Error Level Analysis (ELA) + Noise Forensics
- [ ] **Phase 2**: Pretrained AI-image detector + CLIP cross-modal alignment
- [ ] **Phase 3**: Gemini text claim extraction + Google Search grounding + LIAR FAISS + NLI
- [ ] **Phase 4**: Fusion Engine (rules, conflict thresholds, unverifiable fallback) + Supabase logging
- [ ] **Phase 5**: Frontend Dashboard (React + Vite + Tailwind + Framer Motion)
- [ ] **Phase 6**: Evaluation notebook (test dataset, confusion matrix, error analysis)

For complete technical specifications, JSON contracts, and database schemas, see [SPEC.md](SPEC.md).