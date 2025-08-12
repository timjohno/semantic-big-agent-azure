from typing import Annotated
from semantic_kernel.functions import kernel_function

class InterestRate:
    def __init__(self):
        pass
    @kernel_function(description="Use a model to predict the chance small business has of defaulting on the loan")
    async def interest_rate(
        self,
        loan_data: Annotated[dict, "Structured loan data with fields like loan_amount and organisation_name."],
        risk_score: Annotated[float, "Risk score from a model."],
        survival_prob: Annotated[float, "Survival probability from a model."],
    ) -> dict:
        base_rate = 0.05  # 5% base
        risk_adj = min(risk_score * 0.1, 0.10)  # up to +10%
        survival_adj = max((1 - survival_prob) * 0.05, 0)  # up to +5%
        return {
            "interest_rate": round(base_rate + risk_adj + survival_adj, 4),
        }
