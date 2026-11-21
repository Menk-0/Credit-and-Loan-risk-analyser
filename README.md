# CrediScope – Smart Credit & Loan Risk Analyzer

A machine-learning-based academic project that analyzes borrower profiles, evaluates proposed loan affordability, and generates credit-risk assessment reports.

Built using **Python, FastAPI, Scikit-learn, Pandas, NumPy, HTML, CSS and JavaScript**.

---

# 1. Getting Started – Run the Project

Follow these instructions to run CrediScope on your computer.

## Prerequisites

Install the following:

- Python 3.11
- Git
- pip

## Step 1: Clone the Repository

```bash
git clone https://github.com/Menk-0/Credit-and-Loan-risk-analyser.git
cd Credit-and-Loan-risk-analyser
```

## Step 2: Create a Virtual Environment

A virtual environment keeps project dependencies separate from other Python projects.

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows – PowerShell**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

**Windows – Command Prompt**

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

## Step 3: Install Dependencies

With the virtual environment activated:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Step 4: Start the Application

```bash
uvicorn main:app --reload
```

Wait until the server starts successfully.

Open your browser and visit:

**http://127.0.0.1:8000**

The CrediScope dashboard should now be available.

## Step 5: Run Tests (Optional)

To check whether the application is working correctly:

```bash
pytest -q
```

A successful test run should report:

```text
5 passed
```

The application generates its machine-learning model during startup using the supplied dataset. A previously generated model file does not need to be downloaded separately.

To stop the server, press `Ctrl + C` in the terminal.

---

# 2. Project Overview

CrediScope is designed to demonstrate how machine learning and financial calculations can be combined to analyze borrower profiles and proposed loans.

The application provides two separate assessments:

1. **Borrower Profile Risk Assessment:** Uses a Random Forest machine-learning classifier.
2. **Proposed Loan Risk Assessment:** Uses a transparent, rule-based scoring system.

Keeping these assessments separate helps distinguish historical borrower classification from the affordability of a new loan.

---

# 3. How CrediScope Works

The application follows this general workflow:

```text
User Opens Dashboard
        |
        v
Enters Borrower Information
        |
        v
Enters Proposed Loan Details
        |
        v
FastAPI Receives Input
        |
        v
+---------------------------+
|                           |
v                           v
Random Forest Model    Rule-Based Analysis
|                           |
v                           v
Borrower Profile       Proposed Loan
Risk Assessment        Risk Assessment
|                           |
+-------------+-------------+
              |
              v
      Financial Calculations
              |
              v
       Results & Dashboard
              |
              v
       PDF Report Generation
```

## Step 1: Data Input

The user provides borrower information and proposed loan details through the dashboard.

The inputs include financial information such as:

- Age
- Estimated income
- Credit card balance
- Existing bank loans
- Bank deposits
- Savings and checking accounts
- Properties owned
- Occupation
- Credit and loyalty classifications
- Proposed loan amount
- Interest rate
- Loan tenure
- Existing monthly debt payments
- Monthly living expenses

## Step 2: Backend Processing

The frontend sends the submitted information to the FastAPI backend.

The backend validates the input and processes it for the appropriate assessment.

## Step 3: Machine Learning Assessment

The borrower information is passed to the trained Random Forest classifier.

The model predicts a historical Risk Weighting class from 1 to 5.

## Step 4: Proposed Loan Assessment

The proposed loan is evaluated using financial ratios and predefined risk rules.

The system calculates affordability indicators and produces a separate score from 0 to 100.

## Step 5: Results

The backend returns the assessment results to the frontend.

The dashboard displays the borrower classification, loan-risk score, financial metrics, and supporting explanations.

## Step 6: Reporting

The user can generate a PDF report containing the assessment results.

---

# 4. Machine Learning Model

## Random Forest Classifier

CrediScope uses a Random Forest Classifier from Scikit-learn.

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees to produce a final classification.

### Training Dataset

The application uses:

`data/Banking.csv`

The dataset contains borrower-related financial and demographic information.

The model uses the dataset's `Risk Weighting` column as its target variable.

### Target Classes

The model predicts one of five historical risk-weighting classes:

- Class 1
- Class 2
- Class 3
- Class 4
- Class 5

These represent the categories available in the original dataset.

### Model Training

When the application starts:

1. The backend loads `Banking.csv`.
2. The data is prepared for model training.
3. The Random Forest classifier is trained.
4. The trained model is saved under `models/risk_model.joblib`.
5. The model is used for subsequent borrower assessments.

The generated model artifact is not required in the GitHub repository because the application can train it from the supplied dataset.

### Model Features

The current model uses 15 features:

1. Age
2. Estimated Income
3. Superannuation Savings
4. Amount of Credit Cards
5. Credit Card Balance
6. Bank Loans
7. Bank Deposits
8. Checking Accounts
9. Saving Accounts
10. Foreign Currency Account
11. Business Lending
12. Properties Owned
13. Occupation
14. Fee Structure
15. Loyalty Classification

---

# 5. Proposed Loan Risk Assessment

Unlike the borrower-profile assessment, proposed loan risk is calculated using predefined financial rules.

It produces a score between 0 and 100.

The score evaluates several factors:

| Factor | Risk condition |
|---|---|
| Debt-to-Income Ratio | Above 35% |
| Disposable Income | Below zero |
| EMI-to-Income Ratio | Above 30% |
| Loan-to-Income Ratio | Above 3 times annual income |
| Interest Rate | Above 18% |

Higher penalties are applied to more severe conditions.

The resulting score is classified into four categories:

| Score | Assessment |
|---|---|
| 80–100 | Lower repayment pressure |
| 60–79 | Moderate repayment pressure |
| 40–59 | Elevated repayment pressure |
| Below 40 | High repayment pressure |

The score is an academic screening indicator, not a probability of default.

---

# 6. Financial Calculations

CrediScope includes additional financial calculations to help evaluate the proposed loan.

### EMI Calculation

The system calculates the monthly loan installment using the loan amount, annual interest rate and repayment tenure.

### Debt-to-Income Ratio (DTI)

Measures existing monthly debt obligations relative to monthly income.

### EMI-to-Income Ratio

Measures the proposed monthly installment relative to monthly income.

### Disposable Income

Estimates the amount remaining after accounting for loan installments, existing debt payments and monthly living expenses.

### Expected Loss

The application uses illustrative Probability of Default (PD) proxy and Loss Given Default (LGD) assumptions.

Expected Loss is calculated as:

**Expected Loss = PD × LGD × Loan Amount**

### RAROC

Risk-Adjusted Return on Capital (RAROC) is estimated using simplified profitability and economic-capital assumptions.

These calculations are intended for academic demonstration and are not validated lending models.

---

# 7. Dashboard Features

CrediScope provides a responsive dark-themed dashboard.

### Main Features

- Borrower profile risk classification.
- Proposed loan risk score.
- Loan affordability analysis.
- EMI and amortization calculations.
- Expected loss estimation.
- Indicative profitability and RAROC.
- Portfolio risk distribution charts.
- Risk class distribution visualization.
- Model diagnostics.
- Global permutation feature importance.
- Local one-feature sensitivity analysis.
- PDF assessment report generation.
- Borrower comparison.
- Clickable explanations for important financial metrics.

The charts are rendered without requiring an external chart CDN.

---

# 8. Project Structure

```text
CrediScope/
│
├── main.py
│
├── static/
│   └── index.html
│
├── data/
│   └── Banking.csv
│
├── models/
│   └── risk_model.joblib
│       (Generated automatically)
│
├── test_app.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

### File Responsibilities

| File | Purpose |
|---|---|
| `main.py` | FastAPI backend, model training and financial calculations |
| `static/index.html` | Frontend dashboard, styling and browser-side interactions |
| `data/Banking.csv` | Dataset used for model training |
| `models/` | Stores the generated trained model |
| `test_app.py` | Automated application tests |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes virtual environments, caches and generated files |

---

# 9. API Endpoints

The FastAPI backend exposes the following endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Loads the dashboard |
| GET | `/api/health` | Checks application health |
| GET | `/api/portfolio` | Returns historical dataset portfolio information |
| POST | `/api/analyze` | Performs borrower and proposed loan assessments |
| POST | `/api/report` | Generates a PDF assessment report |

The API can also be explored through FastAPI's interactive documentation:

**http://127.0.0.1:8000/docs**

---

# 10. Configuration

## Currency Configuration

User-facing loan inputs, calculations and PDF report amounts are displayed in Indian Rupees (INR).

The existing dataset's monetary-unit assumptions must be verified against its original data dictionary.

The application uses the `USD_INR_RATE` environment variable for converting presumed USD-denominated dataset features.

The default reference value is `92.5786`.

If the source dataset is confirmed to use USD, a different rate can be configured.

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

Replace the example rate with a verified exchange rate when required.

This conversion is intended to align the presumed dataset scale with user inputs. Loan calculations and reports remain in INR.

---

# 11. Testing

The project includes automated tests using Pytest and FastAPI TestClient.

Run:

```bash
pytest -q
```

The current test suite covers:

- Dashboard and health endpoint.
- Portfolio distribution.
- Borrower and proposed loan analysis.
- PDF report generation.
- Sensitivity analysis.

A successful run currently reports five passing tests.

---

# 12. Important Model Limitations

CrediScope is an academic prototype. Its results should be interpreted carefully.

### 1. Historical Risk Weighting Is Not Loan Default

The dataset's target is `Risk Weighting`, not observed loan-default outcomes.

Therefore, the Random Forest model predicts historical risk-weighting classes rather than actual default probabilities.

### 2. Rule-Based Loan Scoring

The proposed loan score uses predefined thresholds. These thresholds are illustrative and have not been validated against real repayment outcomes.

### 3. Probability of Default

The PD mapping used by the application is an illustrative proxy.

The Random Forest class probabilities are relative class scores and are not calibrated probabilities of default.

### 4. Financial Assumptions

LGD, funding cost, fees, operating costs, economic capital and decision thresholds are simplified academic assumptions.

### 5. Currency Assumptions

The monetary units in the original dataset should be verified before relying on the currency conversion.

### Intended Use

**CrediScope is developed for educational and academic demonstration purposes only.**

It must not be used for actual lending, credit approval, financial decisions or adverse borrower decisions.

---

# 13. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.11 | Main programming language |
| FastAPI | Backend and API development |
| Scikit-learn | Machine learning |
| Random Forest | Classification algorithm |
| Pandas | Dataset processing |
| NumPy | Numerical operations |
| HTML | Frontend structure |
| CSS | Dashboard styling |
| JavaScript | Frontend interactions |
| ReportLab | PDF report generation |
| Pytest | Automated testing |
| Git & GitHub | Version control and collaboration |

---

# 14. Future Improvements

Potential future development includes:

- Training on datasets with actual loan-default labels.
- Comparing Random Forest, XGBoost and CatBoost.
- Calibrated default probability estimation.
- SHAP-based model explainability.
- Improved model evaluation and validation.
- More comprehensive financial risk analysis.
- Additional borrower and loan comparison features.

---

**CrediScope – Smart Credit & Loan Risk Analyzer**

An academic project demonstrating the integration of machine learning, financial calculations, risk assessment and interactive dashboard development.