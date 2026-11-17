# Play Tennis? - Streamlit app

A decision tree (entropy) trained on the classic 14-day tennis dataset,
served as a small Streamlit web app.

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
python train.py                 # creates tennis_model.joblib
streamlit run app.py
```

## Files

| File | What it is |
|------|------------|
| `app.py` | the Streamlit app |
| `train.py` | trains the pipeline and saves `tennis_model.joblib` |
| `tennis.csv` | the dataset |
| `requirements.txt` | the packages (and versions) the app needs |
