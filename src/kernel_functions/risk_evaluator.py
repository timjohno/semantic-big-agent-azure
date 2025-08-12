import streamlit as st
from typing import Annotated
from semantic_kernel.functions import kernel_function

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
        
        return {
            "risk_score": 0.4,
            "service_used": self.runtime,
            "model_used": self.endpoint_name
        }
