# TTA Lab — Test-Time Adaptation in Deep Learning

An interactive research demonstrator based on the uploaded **Test-Time Adaptation in Deep Learning** report and presentation. It reproduces the report's TinyCNN + TENT workflow and adds an interactive dashboard, controlled experiment settings, a BatchNorm-only ablation, image upload prediction, training controls, confusion matrix, and CSV export.

## Features

- **Research dashboard:** dataset, architecture, parameter count and training history.
- **Reproducible experiment:** scikit-learn Digits dataset, stratified 75:25 split, Gaussian noise, clean/noisy/TENT evaluation.
- **TENT adaptation:** only BatchNorm scale (γ) and shift (β) are trainable; target labels are not passed into adaptation.
- **BatchNorm-only ablation:** uses target batch statistics without entropy-gradient updates. This is included to help investigate the attribution caveat in the report; it is not presented as a measured result from the original run.
- **Experiment controls:** noise standard deviation, batch size, TENT learning rate, noise seed, optional confidence gate.
- **Image prediction:** upload a handwritten digit image and see predicted class, confidence, entropy and class probabilities.
- **Training controls:** retrain the TinyCNN with editable epochs, learning rate, batch size and seed.
- **Visual analytics:** method comparison, training-loss chart, entropy-by-batch chart, confusion matrix and prediction samples.
- **CSV export:** download the latest experiment's measured results.

## Report-aligned default configuration

| Setting | Value |
|---|---:|
| Dataset | scikit-learn Digits |
| Total images | 1,797 |
| Train / test | 1,347 / 450 |
| Input | 3 × 32 × 32 |
| TinyCNN parameters | 24,346 |
| Source training | Adam, 0.003, batch 128, 8 epochs |
| Noise | Gaussian σ = 0.30, clipped to [0, 1] |
| Noise seed | 43 |
| TENT learning rate | 0.0001 |
| TENT updates | BatchNorm γ and β only |

The report's single-run reference values are **94.89% clean**, **16.67% noisy source-only**, and **25.78% noisy + TENT**. Your local run may differ slightly due to software versions and execution details. The app computes its own results live; it does not hard-code these reference values.

## Run locally

### Windows
Double-click `run.bat`, or run:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cd backend
python -m uvicorn main:app --reload
```

### macOS / Linux

```bash
chmod +x run.sh
./run.sh
```

Open **http://127.0.0.1:8000**. API documentation is at **http://127.0.0.1:8000/docs**.

First startup trains the small CNN and stores its weights in `artifacts/tinycnn.pt`. The `artifacts/` directory is ignored by Git; the model is regenerated automatically when the app starts if the weights are absent.

## GitHub upload

1. Extract this ZIP.
2. Create a new empty GitHub repository.
3. Upload the extracted project files (not the ZIP itself), or use Git:

```bash
git init
git add .
git commit -m "Build interactive Test-Time Adaptation lab"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

## Important research notes

- The adaptation endpoint deliberately accepts **only images/configuration**, never target labels. Labels are used after adaptation solely to calculate evaluation metrics.
- TENT updates the 224 BatchNorm γ/β parameters (about 0.92% of the 24,346-parameter network); convolution and classifier weights stay frozen.
- Entropy is a confidence/uncertainty measure, not a correctness guarantee. The optional confidence gate is an experimental safeguard, not part of the original report's baseline TENT configuration.
- The report describes TTT and prompt tuning as background methods, not implementations. This repository marks them as theory-only.
- The report notes a single seed, one corruption type and a small dataset. Results should be interpreted as a reproducible teaching case study, not a general robustness claim.
- The BatchNorm-only row is an additional ablation implemented in this app. It should be rerun and reported as a new experiment, not represented as an original report result.

## Project structure

```text
tta-lab/
├── backend/
│   ├── main.py
│   ├── engine.py
│   └── model.py
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── artifacts/
├── requirements.txt
├── run.bat
├── run.sh
├── .gitignore
└── README.md
```

## Stack

Python · PyTorch · FastAPI · scikit-learn · NumPy · HTML · CSS · JavaScript · Chart.js (CDN)
