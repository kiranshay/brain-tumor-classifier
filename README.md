# Brain Tumor MRI Classifier

Full-stack web app for classifying brain MRI scans into four categories: **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**.

Built with EfficientNet-B0 (transfer learning), FastAPI, Supabase, and vanilla JavaScript.

## Architecture

```
Frontend (GitHub Pages)  -->  FastAPI Backend (Render)  -->  Supabase (PostgreSQL)
                                    |
                              EfficientNet-B0
                              (PyTorch, CPU)
```

- **Upload** an MRI scan (or use a built-in sample)
- **Classify** with a fine-tuned EfficientNet-B0 model (~95% test accuracy)
- **Store** every prediction in Supabase with confidence scores and a thumbnail
- **Browse** prediction history and aggregate statistics

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | PyTorch, EfficientNet-B0, transfer learning |
| Backend | FastAPI, Uvicorn |
| Database | Supabase (PostgreSQL) |
| Frontend | HTML, CSS, JavaScript |
| Training | Google Colab (GPU) |
| Deployment | Render (backend), GitHub Pages (frontend) |

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle. ~7,000 MRI images across 4 classes with a pre-defined train/test split.

## Project Structure

```
brain-tumor-classifier/
├── training/
│   └── train.ipynb              # Colab training notebook
├── backend/
│   ├── main.py                  # FastAPI endpoints
│   ├── model.py                 # Model loading & inference
│   ├── database.py              # Supabase client
│   ├── schemas.py               # Pydantic models
│   ├── requirements.txt         # CPU-only PyTorch
│   └── tumor_classifier.pth     # Trained weights (~20MB)
├── frontend/
│   ├── index.html               # Single-page app
│   ├── styles.css               # Styling
│   ├── app.js                   # Frontend logic
│   └── assets/sample_mris/      # Demo images
```

## Setup

### 1. Train the model

Open `training/train.ipynb` in Google Colab with a GPU runtime. Follow the cells to download the dataset, train, and export `tumor_classifier.pth`. Place the exported weights in `backend/`.

### 2. Set up Supabase

Create a free project at [supabase.com](https://supabase.com). Run the SQL in the Supabase SQL editor:

```sql
CREATE TABLE predictions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at timestamptz DEFAULT now(),
  predicted_class text NOT NULL,
  confidence float8 NOT NULL,
  all_confidences jsonb NOT NULL,
  thumbnail_base64 text,
  original_filename text,
  inference_time_ms float8
);

ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow anonymous access" ON predictions
  FOR ALL USING (true) WITH CHECK (true);
```

### 3. Run the backend locally

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export SUPABASE_URL="your-project-url"
export SUPABASE_KEY="your-anon-key"

uvicorn main:app --reload --port 8000
```

### 4. Run the frontend locally

Serve `frontend/` with any static server:

```bash
cd frontend
python -m http.server 5500
```

Open `http://localhost:5500` in your browser.

## Deployment

**Backend** (Render free tier):
- Connect the GitHub repo
- Set root directory to `backend`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Add environment variables: `SUPABASE_URL`, `SUPABASE_KEY`

**Frontend** (GitHub Pages):
- In repo settings, set Pages source to `frontend/` on `main` branch
- Update `API_URL` in `frontend/app.js` to your Render URL

## Author

[Kiran Shay](https://kiranshay.github.io) - Johns Hopkins University, CS & Neuroscience
