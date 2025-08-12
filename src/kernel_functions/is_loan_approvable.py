from typing import Annotated
from semantic_kernel.functions import kernel_function

class IsLoanApprovable:
    def __init__(self):
        self.runtime = "testruntime"
        self.endpoint_name = "fraud-detection-xgb-v1-endpoint"

    @kernel_function(description="Should the loan be approved")
    async def is_loan_approvable(
        self,
        loan_data: Annotated[dict, "Structured loan data with fields like loan_amount and organisation_name."],
        risk_score: Annotated[dict, "Risk score from a model."],
        survival_prob: Annotated[dict, "Survival probability from a model."],
    ) -> dict:
        if risk_score['probability'] > 0.5 or survival_prob['probability'] < 0.5:
            return {
                "verdict": False
            }
        if loan_data['loan_amount'] > 100_000 and risk_score > 0.3:
            return {
                "verdict": False
            }
        return {
            "verdict": True
        }

