from pathlib import Path
from io import BytesIO
from typing import Literal
from datetime import datetime
import os
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Banking.csv"
MODEL_PATH = ROOT / "models" / "risk_model.joblib"
TARGET = "Risk Weighting"
NUMERIC = ["Age", "Estimated Income", "Superannuation Savings", "Amount of Credit Cards",
           "Credit Card Balance", "Bank Loans", "Bank Deposits", "Checking Accounts",
           "Saving Accounts", "Foreign Currency Account", "Business Lending", "Properties Owned"]
CATEGORICAL = ["Occupation", "Fee Structure", "Loyalty Classification"]
FEATURES = NUMERIC + CATEGORICAL
# Reference USD/INR mid-market rate retrieved 20 Nov 2026. Override with USD_INR_RATE env var.
USD_INR_RATE = float(os.getenv("USD_INR_RATE", "92.5786"))
MONEY_FEATURES = ["Estimated Income", "Superannuation Savings", "Credit Card Balance", "Bank Loans",
                  "Bank Deposits", "Checking Accounts", "Saving Accounts",
                  "Foreign Currency Account", "Business Lending"]

app = FastAPI(title="Smart Credit & Loan Risk Analyzer", version="2.0.0")
(ROOT / "static").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


def train_model():
    df = pd.read_csv(DATA_PATH).dropna(subset=[TARGET]).copy()
    X, y = df[FEATURES], df[TARGET].astype(int)
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", num_pipe, NUMERIC), ("cat", cat_pipe, CATEGORICAL)])
    pipe = Pipeline([("prep", prep), ("model", RandomForestClassifier(
        n_estimators=350, min_samples_leaf=2, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    ))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=sorted(y.unique())).tolist()
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        "test_rows": int(len(y_test)), "training_rows": int(len(y_train)),
        "classes": sorted(int(v) for v in y.unique()),
        "per_class": {str(k): {"precision": round(v["precision"], 3), "recall": round(v["recall"], 3),
                               "f1": round(v["f1-score"], 3), "support": int(v["support"])}
                      for k, v in report.items() if str(k).isdigit()},
        "confusion_matrix": cm
    }
    # Permutation importance is computed on held-out raw columns, so it is model-agnostic.
    perm = permutation_importance(pipe, X_test, y_test, n_repeats=3, random_state=42,
                                  scoring="f1_macro", n_jobs=-1)
    importance = sorted(
        [{"feature": col, "importance": round(float(val), 5)}
         for col, val in zip(FEATURES, perm.importances_mean)],
        key=lambda x: x["importance"], reverse=True
    )[:10]
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    return pipe, df, metrics, importance


model, portfolio, model_metrics, feature_importance = train_model()


class Applicant(BaseModel):
    age: int = Field(ge=18, le=100)
    estimated_income: float = Field(gt=0, description="Annual gross income in INR")
    superannuation_savings: float = Field(ge=0)
    credit_cards: int = Field(ge=0, le=30)
    credit_card_balance: float = Field(ge=0)
    bank_loans: float = Field(ge=0)
    bank_deposits: float = Field(ge=0)
    checking_accounts: float = Field(ge=0)
    saving_accounts: float = Field(ge=0)
    foreign_currency_account: float = Field(ge=0)
    business_lending: float = Field(ge=0)
    properties_owned: int = Field(ge=0, le=100)
    occupation: str = "Other"
    fee_structure: Literal["Low", "Mid", "High"] = "Mid"
    loyalty_classification: Literal["Jade", "Silver", "Gold", "Platinum"] = "Silver"
    loan_amount: float = Field(gt=0, description="Loan principal in INR")
    interest_rate: float = Field(gt=0, le=50)
    tenure_months: int = Field(ge=1, le=360)
    loan_type: Literal["Home", "Auto", "Education", "Business", "Personal"] = "Personal"
    monthly_existing_debt_payments: float = Field(default=0, ge=0)
    monthly_living_expenses: float = Field(default=0, ge=0)


LGD = {"Home": .25, "Auto": .35, "Education": .45, "Business": .50, "Personal": .55}
COF = {"Home": 7.5, "Auto": 8.0, "Education": 8.5, "Business": 9.0, "Personal": 9.5}
PD_PROXY = {1: .02, 2: .05, 3: .10, 4: .20, 5: .35}


def amortization(loan: float, annual_rate: float, months: int, cof_annual: float):
    r = annual_rate / 1200
    emi = loan * r / (1 - (1 + r) ** (-months)) if r else loan / months
    balance = loan
    total_interest = 0.0
    total_funding_cost = 0.0
    rows = []
    for month in range(1, months + 1):
        interest = balance * r
        principal = min(balance, emi - interest)
        payment = principal + interest
        funding = balance * (cof_annual / 1200)
        balance = max(0.0, balance - principal)
        total_interest += interest
        total_funding_cost += funding
        if month <= 12 or month == months:
            rows.append({"month": month, "opening_balance": round(balance + principal, 2),
                         "emi": round(payment, 2), "principal": round(principal, 2),
                         "interest": round(interest, 2), "closing_balance": round(balance, 2)})
    return emi, total_interest, total_funding_cost, rows


def analyze_applicant(a: Applicant):
    row = {
        "Age": a.age, "Estimated Income": a.estimated_income,
        "Superannuation Savings": a.superannuation_savings,
        "Amount of Credit Cards": a.credit_cards, "Credit Card Balance": a.credit_card_balance,
        "Bank Loans": a.bank_loans, "Bank Deposits": a.bank_deposits,
        "Checking Accounts": a.checking_accounts, "Saving Accounts": a.saving_accounts,
        "Foreign Currency Account": a.foreign_currency_account,
        "Business Lending": a.business_lending, "Properties Owned": a.properties_owned,
        "Occupation": a.occupation, "Fee Structure": a.fee_structure,
        "Loyalty Classification": a.loyalty_classification
    }
    # The dataset documentation labels Estimated Income in USD; this project assumes the other monetary balance fields use the same unit. User inputs are INR.
    original_row = row.copy()  # applicant's own INR values, kept for display
    # Convert only the model feature row to USD. Loan calculations stay in INR.
    for col in MONEY_FEATURES:
        row[col] = row[col] / USD_INR_RATE
    X = pd.DataFrame([row], columns=FEATURES)
    risk_class = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]
    classes = model.named_steps["model"].classes_
    class_probs = {str(int(c)): round(float(p), 4) for c, p in zip(classes, probs)}
    # Local one-feature-at-a-time sensitivity: replace each feature with its training median/mode
    # and observe the change in probability for the originally predicted class.
    predicted_index = list(classes).index(risk_class)
    current_probability = float(probs[predicted_index])
    local_sensitivity = []
    for feature in FEATURES:
        changed = row.copy()
        if feature in NUMERIC:
            baseline = float(portfolio[feature].median())
        else:
            baseline = str(portfolio[feature].mode(dropna=True).iloc[0])
        changed[feature] = baseline
        changed_probs = model.predict_proba(pd.DataFrame([changed], columns=FEATURES))[0]
        delta_pp = (current_probability - float(changed_probs[predicted_index])) * 100
        current_display = original_row[feature]
        if feature in MONEY_FEATURES:
            current_display = round(float(current_display), 2)
        baseline_display = baseline * USD_INR_RATE if feature in MONEY_FEATURES else baseline
        local_sensitivity.append({"feature": feature, "current_value": current_display,
                                  "baseline_value": round(float(baseline_display), 2) if feature in MONEY_FEATURES else baseline,
                                  "probability_change_pp": round(delta_pp, 2)})
    local_sensitivity.sort(key=lambda item: abs(item["probability_change_pp"]), reverse=True)
    pd_proxy = PD_PROXY.get(risk_class, .10)
    lgd, cof_rate = LGD[a.loan_type], COF[a.loan_type]
    emi, interest_income, funding_cost, schedule = amortization(
        a.loan_amount, a.interest_rate, a.tenure_months, cof_rate)
    fees = a.loan_amount * .01
    operating_cost = a.loan_amount * .005
    expected_loss = pd_proxy * lgd * a.loan_amount
    net_profit = interest_income + fees - funding_cost - operating_cost - expected_loss
    economic_capital = max(a.loan_amount * (np.sqrt(pd_proxy) * .12 + .04), expected_loss)
    raroc = net_profit / economic_capital * 100 if economic_capital else 0.0
    monthly_income = a.estimated_income / 12
    total_monthly_debt = a.monthly_existing_debt_payments + emi
    dti = total_monthly_debt / monthly_income * 100 if monthly_income else 0
    emi_ratio = emi / monthly_income * 100 if monthly_income else 0
    disposable = monthly_income - a.monthly_existing_debt_payments - emi - a.monthly_living_expenses
    if dti <= 35 and disposable >= 0:
        affordability = "Comfortable (indicative)"
    elif dti <= 50 and disposable >= 0:
        affordability = "Stretched (indicative)"
    else:
        affordability = "High repayment pressure (indicative)"
    # Separate proposed-loan affordability engine. This is a transparent rule score,
    # not a second ML model and not a probability of default.
    loan_score = 100
    loan_factors = []
    def apply_loan_rule(name, condition, penalty, explanation):
        nonlocal loan_score
        if condition:
            loan_score -= penalty
            loan_factors.append({"factor": name, "impact": -penalty, "explanation": explanation})
        else:
            loan_factors.append({"factor": name, "impact": 0, "explanation": "No penalty under this demo rule"})
    apply_loan_rule("Debt-to-income", dti > 35, 25 if dti > 50 else 15,
                    f"DTI is {dti:.1f}%; demo threshold is 35% (severe above 50%)")
    apply_loan_rule("Disposable income", disposable < 0, 30,
                    f"Monthly disposable income is ₹{disposable:,.2f}")
    apply_loan_rule("EMI-to-income", emi_ratio > 30, 15,
                    f"New EMI uses {emi_ratio:.1f}% of gross monthly income")
    loan_to_annual_income = a.loan_amount / a.estimated_income
    apply_loan_rule("Loan amount / annual income", loan_to_annual_income > 3, 15,
                    f"Requested principal is {loan_to_annual_income:.2f}× annual gross income")
    apply_loan_rule("Interest rate", a.interest_rate > 18, 10,
                    f"Quoted annual rate is {a.interest_rate:.1f}%")
    loan_score = max(0, min(100, loan_score))
    if loan_score >= 80:
        loan_band = "Lower repayment pressure"
    elif loan_score >= 60:
        loan_band = "Moderate repayment pressure"
    elif loan_score >= 40:
        loan_band = "Elevated repayment pressure"
    else:
        loan_band = "High repayment pressure"

    pd_gate, raroc_gate = pd_proxy < .10, raroc >= 15
    affordability_gate = dti <= 50 and disposable >= 0
    if not affordability_gate:
        decision, action = "REJECT", "Indicative: repayment burden exceeds the demo affordability limits"
    elif pd_gate and raroc_gate:
        decision, action = "APPROVE", "Indicative: standard terms may be reviewed"
    elif pd_gate and not raroc_gate:
        decision, action = "REPRICE", "Indicative: review pricing or costs"
    elif not pd_gate and raroc_gate:
        decision, action = "PRICE PREMIUM", "Indicative: elevated risk; review pricing and policy"
    else:
        decision, action = "REJECT", "Indicative: elevated risk and insufficient risk-adjusted return"
    return {
        "borrower_profile_assessment": {
            "risk_class": risk_class,
            "risk_class_label": ["", "Very Low", "Low", "Moderate", "High", "Very High"][risk_class],
            "class_probabilities": class_probs,
            "target": "Historical dataset Risk Weighting (classes 1–5)",
            "meaning": "Compares the applicant profile with the dataset's Risk Weighting labels; not a default forecast."
        },
        "proposed_loan_assessment": {
            "score": loan_score,
            "band": loan_band,
            "basis": "Transparent affordability and loan-structure rules; not machine learning or probability of default.",
            "factors": loan_factors,
            "dti": round(dti, 2),
            "emi_to_income": round(emi_ratio, 2),
            "disposable_income": round(disposable, 2),
            "loan_to_annual_income": round(loan_to_annual_income, 3)
        },
        "risk_class": risk_class,
        "risk_class_label": ["", "Very Low", "Low", "Moderate", "High", "Very High"][risk_class],
        "class_probabilities": class_probs, "local_sensitivity": local_sensitivity[:8], "pd_estimate": round(pd_proxy, 4), "pd_is_proxy": True,
        "emi": round(float(emi), 2), "lgd": lgd,
        "expected_loss": round(float(expected_loss), 2), "interest_income": round(float(interest_income), 2),
        "fees": round(float(fees), 2), "cost_of_funds": round(float(funding_cost), 2),
        "operating_cost": round(float(operating_cost), 2), "net_profit": round(float(net_profit), 2),
        "economic_capital": round(float(economic_capital), 2), "raroc": round(float(raroc), 2),
        "monthly_income": round(monthly_income, 2), "monthly_existing_debt_payments": a.monthly_existing_debt_payments,
        "monthly_living_expenses": a.monthly_living_expenses,
        "total_monthly_debt_payments": round(total_monthly_debt, 2),
        "dti": round(dti, 2), "emi_to_income": round(emi_ratio, 2),
        "disposable_income": round(disposable, 2), "affordability": affordability,
        "total_repayment": round(float(emi * a.tenure_months), 2),
        "total_interest": round(float(interest_income), 2), "amortization_schedule": schedule,
        "decision": decision, "action": action, "model_metrics": model_metrics,
        "fx_rate_usd_inr": USD_INR_RATE, "disclaimer": "Academic demonstration only. Borrower Profile Risk predicts the dataset Risk Weighting label, not default. Proposed Loan Risk is a transparent rule-based affordability score, not a trained default model. PD, LGD, pricing, fees, operating costs and thresholds are illustrative, not lending policy."
    }


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": "RandomForestClassifier", "dataset_rows": int(len(portfolio)),
            "target": TARGET, "metrics": model_metrics, "feature_importance": feature_importance,
            "currency": "INR", "dataset_currency": "USD (assumed for monetary fields)", "usd_inr_rate": USD_INR_RATE,
            "fx_rate_as_of": os.getenv("USD_INR_RATE_AS_OF", "2026-11-20 (configurable reference)"),
            "dataset_currency_note": f"User enters INR. Monetary model features are converted to USD at ₹{USD_INR_RATE:.4f} per USD. Update USD_INR_RATE in the environment when the reference rate changes."}


@app.get("/api/portfolio")
def portfolio_summary():
    dist = portfolio[TARGET].value_counts().sort_index()
    counts = {str(int(k)): int(v) for k, v in dist.items()}
    total = max(1, sum(counts.values()))
    return {"records": int(len(portfolio)), "risk_distribution": counts,
            "risk_percentages": {k: round(v / total * 100, 2) for k, v in counts.items()},
            "risk_target_note": "Risk Weighting is the dataset target; it is not a verified loan-default outcome."}


@app.post("/api/analyze")
def analyze(a: Applicant):
    return analyze_applicant(a)


@app.post("/api/report")
def report(a: Applicant):
    result = analyze_applicant(a)
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except ImportError as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="PDF dependency missing. Install it with: python -m pip install reportlab") from exc
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=17*mm, leftMargin=17*mm,
                            topMargin=16*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallNote", parent=styles["BodyText"], fontSize=8, leading=11,
                              textColor=colors.HexColor("#555555")))
    story = [Paragraph("Smart Credit &amp; Loan Risk Analyzer", styles["Title"]),
             Paragraph(f"Academic assessment report · {datetime.now().strftime('%d %b %Y, %I:%M %p')}", styles["Normal"]),
             Spacer(1, 8*mm), Paragraph("Risk assessment", styles["Heading2"])]
    def indian_amount(value):
        value = float(value)
        whole, frac = f"{abs(value):.2f}".split(".")
        if len(whole) > 3:
            tail, head = whole[-3:], whole[:-3]
            pairs = []
            while head:
                pairs.insert(0, head[-2:])
                head = head[:-2]
            whole = ",".join(pairs + [tail])
        return ("-" if value < 0 else "") + "INR " + whole + "." + frac
    money = indian_amount
    rows = [["Metric", "Result"], ["Risk weighting class", f"{result['risk_class']} · {result['risk_class_label']}"],
            ["Illustrative PD proxy", f"{result['pd_estimate']*100:.1f}%"], ["Monthly EMI", money(result["emi"])],
            ["Expected loss", money(result["expected_loss"])], ["RAROC", f"{result['raroc']:.2f}%"],
            ["Debt-to-income ratio", f"{result['dti']:.2f}%"], ["EMI-to-income ratio", f"{result['emi_to_income']:.2f}%"],
            ["Disposable income after listed costs", money(result["disposable_income"])],
            ["Affordability indicator", result["affordability"]], ["Indicative rule output", result["decision"]]]
    table = Table(rows, colWidths=[85*mm, 80*mm])
    table.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0),colors.HexColor("#0b1930")),
                               ("TEXTCOLOR", (0,0),(-1,0),colors.white), ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#dce3ec")),
                               ("PADDING",(0,0),(-1,-1),7), ("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story += [table, Spacer(1, 7*mm), Paragraph("Borrower and loan inputs", styles["Heading2"])]
    input_rows = [["Input", "Value"], ["Age", str(a.age)], ["Annual income", money(a.estimated_income)],
                  ["Existing monthly debt payments", money(a.monthly_existing_debt_payments)],
                  ["Monthly living expenses", money(a.monthly_living_expenses)], ["Loan amount", money(a.loan_amount)],
                  ["Interest rate", f"{a.interest_rate:.2f}% p.a."], ["Tenure", f"{a.tenure_months} months"],
                  ["Loan type", a.loan_type]]
    it = Table(input_rows, colWidths=[85*mm,80*mm]); it.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#245b91")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#dce3ec")),("PADDING",(0,0),(-1,-1),6)]))
    story += [it, Spacer(1, 7*mm), Paragraph("Financial breakdown", styles["Heading2"])]
    fin = [["Component", "Amount"]] + [[k, money(result[v])] for k,v in [
        ("Total interest income", "interest_income"), ("Fees", "fees"), ("Cost of funds", "cost_of_funds"),
        ("Operating cost", "operating_cost"), ("Expected loss", "expected_loss"),
        ("Net profit (simplified)", "net_profit"), ("Economic capital", "economic_capital"),
        ("Total repayment", "total_repayment")]]
    ft = Table(fin, colWidths=[85*mm,80*mm]); ft.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#245b91")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#dce3ec")),("PADDING",(0,0),(-1,-1),6)]))
    story += [ft, Spacer(1, 7*mm), Paragraph("Methodology and limitations", styles["Heading2"]),
              Paragraph(result["disclaimer"], styles["SmallNote"]),
              Paragraph(f"The source dataset documentation labels Estimated Income as USD; the project assumes the other monetary balance fields use the same unit. Applicant inputs and financial outputs are INR; monetary model features are converted to USD using the configured reference rate of INR {USD_INR_RATE:.4f} per USD. This report is for academic demonstration only and is not a lending decision.", styles["SmallNote"])]
    try:
        doc.build(story)
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {type(exc).__name__}: {exc}") from exc
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf",
                             headers={"Content-Disposition": "attachment; filename=credit-risk-assessment.pdf"})
