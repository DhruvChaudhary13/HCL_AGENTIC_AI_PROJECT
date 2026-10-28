# Patient Health Outcome Predictor

A small, deployable web app that estimates the risk of a bad health outcome (by default: **hospital readmission within 30 days**) from a patient's demographics, medical history and treatment details.

You train the model yourself in a Jupyter notebook. The web app then loads the saved model and builds its input form from it.

> **Demo / learning project.** Not a medical device. Do not use it for clinical decisions.

## Structure

```
patient-outcome-predictor/
├── notebook/
│   ├── train_model.ipynb     Step-by-step training notebook (you run this)
│   ├── requirements.txt      Libraries for the notebook
│   └── data/                 Put the dataset CSV here
├── backend/
│   ├── app.py                Flask app: JSON API + serves the frontend
│   ├── predictor.py          Loads the trained model, validates input, predicts
│   └── model/                The notebook saves outcome_model.joblib here
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js             Builds the form from /api/schema, calls /api/predict
├── requirements.txt
├── Dockerfile
├── Procfile                  Heroku / Railway
├── render.yaml               Render
└── README.md
```

One Flask process serves both the API (`/api/*`) and the static frontend, so there is a single thing to deploy.

## Quick start

**1. Install**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r notebook/requirements.txt
```

**2. Train the model**

```bash
cd notebook
jupyter lab
```

Open `train_model.ipynb` and run all cells from top to bottom. It downloads the dataset (or tells you how to do it by hand), cleans it, compares two models, evaluates on a hidden test set, and saves `backend/model/outcome_model.joblib`.

At the end of Step 12 the notebook prints a line like `scikit-learn==1.5.2`. Put that exact line in the root `requirements.txt` so the server uses the same version that created the model.

**3. Run the app**

```bash
cd ..                               # back to the project root
python backend/app.py
```

Open http://localhost:5000. If you start the app before training, the page shows a "Model not ready" message instead of the form.

## Datasets you can use

| Dataset | Size | Outcome to predict | Fit for this project |
|---|---|---|---|
| [**Diabetes 130-US Hospitals (1999-2008)**](https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008), UCI | 101,766 hospital visits | Readmission within 30 days | **Best match and the notebook default.** Has demographics, diagnoses, medications, procedures and prior visits. CC BY 4.0. |
| [Heart Failure Clinical Records](https://archive.ics.uci.edu/dataset/519/heart-failure-clinical-records), UCI | 299 patients | Death during follow-up | Tiny and quick to try. Good for learning, too small for a reliable model. CC BY 4.0. Do not use its `time` column as an input (it leaks the answer). |
| [MIMIC-IV Clinical Database Demo](https://www.physionet.org/content/mimic-iv-demo/), PhysioNet | 100 patients | You define it (e.g. in-hospital death, length of stay) | Real hospital tables (admissions, diagnoses, prescriptions, labs), open access, but you must join tables and build features yourself, and 100 patients is too few to train on. The full MIMIC-IV is much larger but needs a credentialed PhysioNet account (training course + data use agreement). |
| [Synthea](https://github.com/synthetichealth/synthea/wiki/DPG) synthetic patients | Any size | You define it | Open-source generator of realistic *fake* patient records (conditions, medications, procedures). No privacy issues, public domain data. Good for testing the pipeline, but the patterns are not real clinical evidence. |

Kaggle also hosts many small health datasets (for example stroke or heart-disease prediction). They need a free account, and you should check each dataset's licence before using it.

### Using a different dataset

Open the notebook and edit **Step 2 (Settings)**: the file name, the label column, the outcome name and the feature lists. Replace the cleaning code in **Step 5** with whatever your file needs. Steps 6 to 13 stay the same, and the web form updates by itself because it is built from the saved model.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check (also reports whether the model is loaded) |
| GET | `/api/schema` | Input fields, outcome name, model scores, risk bands. Returns 503 if no model |
| POST | `/api/predict` | Takes the fields from `/api/schema`, returns probability, risk level, percentile and top factors |

`/api/schema` shows the exact field names for your trained model. With the default dataset, a request looks like this:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"time_in_hospital":5,"num_lab_procedures":45,"num_procedures":1,
       "num_medications":16,"number_outpatient":0,"number_emergency":1,
       "number_inpatient":2,"number_diagnoses":9,"age":"[70-80)",
       "gender":"Female","admission_type":"Emergency",
       "primary_diagnosis":"Circulatory","A1Cresult":"Not measured",
       "insulin":"Steady","med_change":"Changed","diabetesMed":"Yes"}'
```

## Deploy

Commit `backend/model/outcome_model.joblib` together with the code. Hosting platforms build from your repository, so the model file must be in it. It is small (a few hundred KB).

**Render:** push the folder to GitHub, create a new Blueprint pointing at the repo. `render.yaml` does the rest.

**Docker** (Fly.io, Cloud Run, a VPS, ...):

```bash
docker build -t outcome-predictor .
docker run -p 8000:8000 outcome-predictor
```

**Heroku / Railway:** the `Procfile` and root `requirements.txt` are detected automatically.

## How the results are explained

- **Probability:** the model's estimated chance of the outcome.
- **Risk level:** relative to the held-out test patients. By default the top 10% of predicted risk is *High*, the next 30% *Moderate*, the rest *Low* (change `RISK_PERCENTILES` in the notebook). The notebook also prints the *actual* outcome rate in each band so you can check that it makes sense.
- **What influenced this result:** for each input, the app swaps it for the typical value and measures how much the probability changes. This is a simple, easy-to-read approximation, not proof of cause and effect.

## Responsible use

Race is intentionally not used as an input in the default setup. Check any model you build for unfair differences between patient groups, and handle real patient data according to the laws that apply to you (for example HIPAA, GDPR, or India's DPDP Act).
