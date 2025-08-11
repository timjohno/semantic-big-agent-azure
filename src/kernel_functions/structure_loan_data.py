from typing import Annotated
from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function

class StructureLoanData:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel

    @kernel_function(description="Return a JSON containing structured loan_data, use before calling other plugins")
    async def StructureLoanData(self, loan_text: Annotated[str, "The unstructured loan_text string input"]) -> str:
        prompt = f"""
Extract the following fields from the text below. If a field is not present, leave it blank. for loan_amount, extract the number. For example USD 150,000,000 should be loan_amount: 150000000

Required fields (as JSON):
{{
    "organisation_name": "",
    "loan_amount": "",
}}

Text:
\"\"\"{loan_text}\"\"\"

Respond ONLY with a valid JSON object. Do not include any text before or after the JSON.
"""
        completion = await self.kernel.invoke_prompt(prompt)
        if hasattr(completion, "result"):
            return str(completion.result).strip()
        return str(completion).strip()
