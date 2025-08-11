from typing import Annotated
import re
from fuzzywuzzy import process
from semantic_kernel.functions import kernel_function
import numpy as np


class MockDatabaseConnector:
    @kernel_function(description="Retrieve data from the database")
    async def retrieve_data(
        self,
        loan_data: Annotated[dict, "Structured loan object containing organisation_name."]
    ) -> dict:
        
        organisation_name = loan_data.get("organisation_name")
        data = self.lookup_or_fallback(organisation_name)
        print('requesting data for:', organisation_name)
        print(data)
        return data

    def lookup_or_fallback(self, company_name, threshold=85):
        # Normalise input
        norm_input = self.normalise_name(company_name)

        # Create a mapping of normalised name → original name
        norm_to_original = {self.normalise_name(name): name for name in PREDEFINED_COMPANIES.keys()}

        # Fuzzy match on normalised names
        best_match, score, _ = process.extractOne(
            norm_input,
            list(norm_to_original.keys()),
    )
        if score >= threshold:
            return PREDEFINED_COMPANIES[best_match]
        else:
            return self.generate_random_datapoint(np.random.choice(["good", "medium", "bad"]))


    @staticmethod
    def normalise_name(name):
        name = name.lower()
        name = re.sub(r'\bltd\b', 'limited', name)
        name = re.sub(r'\bco\b', 'company', name)
        name = re.sub(r'[^a-z0-9 ]+', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    @staticmethod
    def generate_random_datapoint(risk_category: str) -> dict:

        if risk_category not in ["good", "medium", "bad"]:
            raise ValueError("Invalid risk category. Choose from 'good', 'medium', or 'bad'.")

            # Define possible sectors and locations
        SECTORS = ["Retail", "Manufacturing", "Hospitality", "Technology", "Healthcare"]
        LOCATIONS = ["London", "Birmingham", "Leeds", "Manchester", "Glasgow"]
        if risk_category == "good":
            return {
                "ebitda_margin": round(np.random.uniform(0.15, 0.3), 2),
                "debt_to_equity": round(np.random.uniform(0.2, 1.0), 2),
                "current_ratio": round(np.random.uniform(1.5, 3.0), 2),
                "revenue_growth": round(np.random.uniform(0.05, 0.15), 2),
                "loan_amount": np.random.randint(5000, 50000),
                "sector": np.random.choice(SECTORS),
                "location": np.random.choice(LOCATIONS)
            }

        elif risk_category == "medium":
            return {
                "ebitda_margin": round(np.random.uniform(0.08, 0.15), 2),
                "debt_to_equity": round(np.random.uniform(1.0, 2.0), 2),
                "current_ratio": round(np.random.uniform(1.0, 1.5), 2),
                "revenue_growth": round(np.random.uniform(0.0, 0.05), 2),
                "loan_amount": np.random.randint(20000, 100000),
                "sector": np.random.choice(SECTORS),
                "location": np.random.choice(LOCATIONS)
            }

        elif risk_category == "bad":
            return {
                "ebitda_margin": round(np.random.uniform(-0.05, 0.08), 2),
                "debt_to_equity": round(np.random.uniform(2.0, 5.0), 2),
                "current_ratio": round(np.random.uniform(0.5, 1.0), 2),
                "revenue_growth": round(np.random.uniform(-0.1, 0.0), 2),
                "loan_amount": np.random.randint(50000, 200000),
                "sector": np.random.choice(SECTORS),
                "location": np.random.choice(LOCATIONS)
            }


PREDEFINED_COMPANIES = {
    "Greensplice": {
        "ebitda_margin": 0.22,
        "debt_to_equity": 0.5,
        "current_ratio": 2.1,
        "revenue_growth": 0.12,
        "loan_amount": 30000,
        "sector": "Sustainable Manufacturing",
        "location": "Manchester"
    },
    "ByteCrate": {
        "ebitda_margin": 0.12,
        "debt_to_equity": 1.5,
        "current_ratio": 1.2,
        "revenue_growth": 0.03,
        "loan_amount": 75000,
        "sector": "Technology",
        "location": "London"
    },
    "SolarTrack": {
        "ebitda_margin": 0.04,
        "debt_to_equity": 3.5,
        "current_ratio": 0.8,
        "revenue_growth": -0.02,
        "loan_amount": 120000,
        "sector": "Renewable Energy",
        "location": "Birmingham"
    }
}