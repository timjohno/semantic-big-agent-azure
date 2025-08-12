import streamlit as st
from typing import Annotated
from semantic_kernel.functions import kernel_function

class InterestRate:
    def __init__(self):
        self.runtime = "testruntime"
        self.endpoint_name = "fraud-detection-xgb-v1-endpoint"

    @kernel_function(description="Use a model to predict the chance small business has of defaulting on the loan")
    async def interest_rate(
        self,
        loan_data: Annotated[dict, "Structured loan data with fields like loan_amount and organisation_name."],
        risk_score: Annotated[dict, "Risk score from a model."],
        survival_prob: Annotated[dict, "Survival probability from a model."],
    ) -> dict:
        base_rate = 0.05  # 5% base
        risk_adj = min(risk_score['probability'] * 0.1, 0.10)  # up to +10%
        survival_adj = max((1 - survival_prob['probability']) * 0.05, 0)  # up to +5%
        return {
            "interest_rate": round(base_rate + risk_adj + survival_adj, 4),
            "service_used": self.runtime,
            "model_used": self.endpoint_name
        }
