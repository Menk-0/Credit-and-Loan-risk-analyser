from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)
SAMPLE = {
    "age": 35, "estimated_income": 600000, "superannuation_savings": 100000,
    "credit_cards": 2, "credit_card_balance": 25000, "bank_loans": 100000,
    "bank_deposits": 150000, "checking_accounts": 25000, "saving_accounts": 50000,
    "foreign_currency_account": 0, "business_lending": 0, "properties_owned": 1,
    "occupation": "Software Consultant", "fee_structure": "Mid",
    "loyalty_classification": "Silver", "loan_amount": 300000,
    "interest_rate": 12, "tenure_months": 36, "loan_type": "Personal",
    "monthly_existing_debt_payments": 5000, "monthly_living_expenses": 15000
}

def test_home_and_health():
    home = client.get('/')
    assert home.status_code == 200
    assert b'CrediScope' in home.content
    health = client.get('/api/health')
    assert health.status_code == 200
    assert health.json()['currency'] == 'INR'
    assert len(health.json()['feature_importance']) > 0

def test_portfolio_chart_data():
    data = client.get('/api/portfolio').json()
    assert data['records'] > 0
    assert abs(sum(data['risk_percentages'].values()) - 100) < 0.1

def test_analyze_inr_outputs_and_affordability():
    response = client.post('/api/analyze', json=SAMPLE)
    assert response.status_code == 200
    data = response.json()
    assert data['emi'] > 0
    assert data['total_repayment'] > SAMPLE['loan_amount']
    assert data['dti'] > 0
    assert data['fx_rate_usd_inr'] > 0
    assert data['pd_is_proxy'] is True
    assert len(data['local_sensitivity']) > 0
    assert data['total_interest'] > 0
    assert data['borrower_profile_assessment']['target'].startswith('Historical dataset')
    assert 0 <= data['proposed_loan_assessment']['score'] <= 100
    assert data['proposed_loan_assessment']['basis'].startswith('Transparent')

def test_pdf_report():
    response = client.post('/api/report', json=SAMPLE)
    assert response.status_code == 200, response.text
    assert response.content.startswith(b'%PDF')
    assert 'application/pdf' in response.headers['content-type']


def test_sensitivity_current_value_is_in_inr():
    data = client.post('/api/analyze', json=SAMPLE).json()
    income = next(r for r in data['local_sensitivity'] if r['feature'] == 'Estimated Income')
    assert income['current_value'] == SAMPLE['estimated_income']
