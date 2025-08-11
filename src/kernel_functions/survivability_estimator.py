from typing import Annotated
import json
import boto3
from semantic_kernel.functions import kernel_function

#self.runtime = boto3.client("sagemaker-runtime")
#self.endpoint_name = "claim-amount-linear-v2-endpoint"

class SurvivabilityEstimator:
    def __init__(self):
        self.runtime = "testruntime"
        self.endpoint_name = "testendpoint"

    @kernel_function(description="Calculate the survivability of a business, i.e. the length of time they are expected to survive")
    async def estimate_size(
        self,
        claim_data: Annotated[dict, "Structured company data."]
    ) -> dict:
        
        return {
            "years": 2.5,
            "service_used": self.runtime,
            "model_used": self.endpoint_name 
        }

    
'''coverage_amount = claim_data.get("coverage_amount", "")
region_of_operation = claim_data.get("region_of_operation", "").lower()
#coverage_amount_str = claim_data.get("coverage_amount", "").lower()
#region_of_operation = claim_data.get("region_of_operation", "").lower()
#cleaned = coverage_amount_str.lower().replace(",", "")
# Extract digits only
#digits = ''.join(filter(str.isdigit, cleaned))
# Convert to integer and scale down
coverage_amount = int(coverage_amount) // 1000 if coverage_amount else 0

if region_of_operation == "gb":
    region_value = 0
elif region_of_operation == "usa":
    region_value = 1
elif region_of_operation == "eu":
    region_value = 2
elif region_of_operation == "asia":
    region_value = 3
elif region_of_operation == "africa":
    region_value = 4
else:
    region_value = 5

payload = f"{coverage_amount},{region_value}"

#try:
response = self.runtime.invoke_endpoint(
    EndpointName=self.endpoint_name,
    ContentType="text/csv",
    Body=payload
)
result = json.loads(response["Body"].read().decode())
prediction = result["predictions"][0]["score"]

return {
    "estimated_insurance_premium": round(prediction, 2),
    "currency": "GBP",
    "service_used": self.runtime,
    "model_used": self.endpoint_name 
}'''
