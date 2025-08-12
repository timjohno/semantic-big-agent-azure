import math
import numpy as np
from typing import Annotated
from semantic_kernel.functions import kernel_function


def logistic(x):
    return 1 / (1 + math.exp(-x))

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
        financial_data['loan_amount'] = int(loan_data.get('loan_amount', 0))
        prob, flags = self.calculate_probability(financial_data)
        return {
            "probability": prob,
            "flags": flags,
            "service_used": self.runtime,
            "model_used": self.endpoint_name 
        }

    @staticmethod
    def calculate_probability(company_features) -> float:

        score = 0
        flags = {}

        # EBITDA margin effect
        if company_features['ebitda_margin'] > 0.25:
            flags['ebitda_margin'] = "Increase survival: EBITDA margin above 25%"
            score += 2
        elif company_features['ebitda_margin'] > 0.15:
            flags['ebitda_margin'] = "Increase survival: EBITDA margin above 15%"
            score += 1
        else:
            flags['ebitda_margin'] = "Decrease survival: EBITDA margin below 15%"
            score -= 1

        # Debt-to-equity effect
        if company_features['debt_to_equity'] < 1:
            flags['debt_to_equity'] = "Increase survival: Debt-to-equity below 1.0"
            score += 2
        elif company_features['debt_to_equity'] < 2:
            flags['debt_to_equity'] = "Increase survival: Debt-to-equity below 2.0"
            score += 1
        else:
            flags['debt_to_equity'] = "Decrease survival: Debt-to-equity above 2.0"
            score -= 1

        # Current ratio effect
        if company_features['current_ratio'] > 2:
            flags['current_ratio'] = "Increase survival: Current ratio above 2.0"
            score += 2
        elif company_features['current_ratio'] > 1:
            flags['current_ratio'] = "Increase survival: Current ratio above 1.0"
            score += 1
        else:
            flags['current_ratio'] = "Decrease survival: Current ratio below 1.0"
            score -= 1

        # Sector effects
        sector_effects = {
            "Hospitality": -2,
            "Retail": -1,
            "Manufacturing": 0,
            "SaaS": 2,
            "Renewable": 1,
            "Technology": -1,
        }
        sector_adj = sector_effects.get(company_features['sector'], 0)
        score += sector_adj
        flags['sector'] = f"Sector effect: {sector_adj:+d}"

        # Location effects
        location_effects = {
            "London": 1,
            "Manchester": 0,
            "Birmingham": -1,
            "Leeds": 0,
            "Glasgow": -1
        }
        location_adj = location_effects.get(company_features['location'], 0)
        score += location_adj
        flags['location'] = f"Location effect: {location_adj:+d}"

        # Logistic probability
        prob = logistic(score)
        prob = np.clip(prob, 0.1, 0.9)  # Ensure probability is within (0, 1)
        return prob, flags
