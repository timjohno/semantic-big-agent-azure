from typing import Annotated
from semantic_kernel.functions import kernel_function


class SurvivabilityEstimator:
    def __init__(self):
        self.runtime = "testruntime"
        self.endpoint_name = "testendpoint"

    @kernel_function(description="Calculate the 3 year survivability of a business")
    async def estimate_probability(
        self,
        loan_data: Annotated[dict, "Structured loan data with fields like loan_amount and organisation_name."],
        financial_data: Annotated[dict, "Structured financial data of the company."]
    ) -> dict:
        
        return {
            "probability": 0.8,
            "service_used": self.runtime,
            "model_used": self.endpoint_name 
        }
