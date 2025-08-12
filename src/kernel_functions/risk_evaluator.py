import math
from typing import Annotated
from semantic_kernel.functions import kernel_function


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class RiskEvaluator:
    def __init__(self):
        self.runtime = "testruntime"
        self.endpoint_name = "fraud-detection-xgb-v1-endpoint"

    @kernel_function(description="Use a model to predict the chance small business has of defaulting on the loan")
    async def assess_risk(
        self,
        loan_data: Annotated[dict, "Structured loan data with fields like loan_amount and organisation_name."],
        financial_data: Annotated[dict, "Structured financial data of the company."]
    ) -> dict:
        financial_data['loan_amount'] = int(loan_data.get('loan_amount', 0))
        return {
            "risk_score": self.interpret_risk_score(financial_data),
            "service_used": self.runtime,
            "model_used": self.endpoint_name
        }

    @staticmethod
    def interpret_risk_score(company_features):
        """
        Interprets risk score based on simple threshold rules.
        Maps final score to a value between 0 and 1.
        """
        flags = {}
        total_score = 0
        max_score = 5   # Max possible risk points
        min_score = -5  # Min possible risk points

        # EBITDA margin
        if company_features['ebitda_margin'] < 0.1:
            flags['ebitda_margin'] = "Increase risk: EBITDA margin below 10%"
            total_score += 1
        elif company_features['ebitda_margin'] > 0.2:
            flags['ebitda_margin'] = "Decrease risk: EBITDA margin above 20%"
            total_score -= 1
        else:
            flags['ebitda_margin'] = "Neutral: EBITDA margin moderate"

        # Debt-to-equity ratio
        if company_features['debt_to_equity'] > 2.5:
            flags['debt_to_equity'] = "Increase risk: Debt-to-equity above 2.5"
            total_score += 1
        elif company_features['debt_to_equity'] < 1.0:
            flags['debt_to_equity'] = "Decrease risk: Debt-to-equity below 1.0"
            total_score -= 1
        else:
            flags['debt_to_equity'] = "Neutral: Debt-to-equity moderate"

        # Current ratio
        if company_features['current_ratio'] < 1.0:
            flags['current_ratio'] = "Increase risk: Current ratio below 1.0"
            total_score += 1
        elif company_features['current_ratio'] > 2.0:
            flags['current_ratio'] = "Decrease risk: Current ratio above 2.0"
            total_score -= 1
        else:
            flags['current_ratio'] = "Neutral: Current ratio moderate"

        # Revenue growth
        if company_features['revenue_growth'] < 0:
            flags['revenue_growth'] = "Increase risk: Negative revenue growth"
            total_score += 1
        elif company_features['revenue_growth'] > 0.1:
            flags['revenue_growth'] = "Decrease risk: Revenue growth above 10%"
            total_score -= 1
        else:
            flags['revenue_growth'] = "Neutral: Revenue growth moderate"

        # Loan amount
        if company_features['loan_amount'] > 100000:
            flags['loan_amount'] = "Increase risk: Loan amount above £100k"
            total_score += 1
        elif company_features['loan_amount'] < 50000:
            flags['loan_amount'] = "Decrease risk: Loan amount below £50k"
            total_score -= 1
        else:
            flags['loan_amount'] = "Neutral: Loan amount moderate"

        # Normalize score to 0–1
        risk_score_normalized = (total_score - min_score) / (max_score - min_score)

        return risk_score_normalized, flags
