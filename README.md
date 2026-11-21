
# Smart Credit & Loan Risk Analyzer

A local academic prototype using FastAPI, scikit-learn and a responsive dark dashboard. All user-facing money inputs, calculations and PDF report amounts are INR (₹). The existing dataset's monetary-unit assumptions must be checked against its original data dictionary.

## Features

- Borrower Profile Risk: Random Forest classification of the dataset's Risk Weighting label (1–5)
- Proposed Loan Risk: separate, transparent rule-based affordability score using DTI, disposable income, EMI-to-income, loan-to-income and interest-rate checks
- INR loan EMI, amortization schedule, interest income and indicative profitability
- Affordability calculations: DTI, EMI-to-income and disposable income
- Global permutation importance and local one-feature sensitivity
- Portfolio risk donut chart, class distribution bars and model diagnostics
- PDF assessment report and borrower comparison
- Clickable metric explanations; responsive dark UI; no chart CDN dependency

## Run (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

## Run (Windows – PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

## Run (Windows – Command Prompt)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

Open your browser:

**http://127.0.0.1:8000**

### Windows PowerShell Activation Note

If PowerShell blocks virtual environment activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.venv\Scripts\Activate.ps1
```

## Tests

Run on macOS / Linux:

```bash
pytest -q
```

Run on Windows:

```powershell
pytest -q
```

## Optional Exchange-Rate Configuration

The model feature conversion uses `USD_INR_RATE` (default reference value: 92.5786). Set a current, verified rate before running if the source dataset's monetary features are confirmed to be USD.

**macOS / Linux:**

```bash
export USD_INR_RATE=90.50
```

**Windows PowerShell:**

```powershell
$env:USD_INR_RATE="90.50"
```

**Windows Command Prompt:**

```cmd
set USD_INR_RATE=90.50
```

This conversion is only for matching the presumed source-dataset scale. Loan calculations and reports remain INR.

## Important Modelling Limitations

Banking.csv has a Risk Weighting target, not observed loan-default labels. The app predicts risk-weighting classes.

The proposed-loan score is a transparent rule-based screening indicator, not a trained default classifier. Its thresholds are illustrative. The PD mapping, LGD, funding cost, fee, operating cost, economic capital and decision thresholds are also illustrative academic assumptions. The model's class probabilities are not calibrated probabilities of default.

**Do not use this prototype for actual lending, credit approval or adverse decisions.**
