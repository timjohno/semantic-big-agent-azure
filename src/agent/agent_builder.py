import streamlit as st
# Plugin classes
from kernel_functions.mock_database_connector import MockDatabaseConnector
from kernel_functions.risk_evaluator import RiskEvaluator
from kernel_functions.survivability_estimator import SurvivabilityEstimator
from kernel_functions.counterfactual import Counterfactual
from kernel_functions.structure_loan_data import StructureLoanData
from kernel_functions.vector_memory import VectorMemoryRAGPlugin
from kernel_functions.is_loan_approvable import IsLoanApprovable
from kernel_functions.interest_rate import InterestRate

# Semantic Kernel core
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import OpenAIChatPromptExecutionSettings


openai_api_type = "azure"
openai_key = st.secrets["openai"]["AZURE_OPENAI_API_KEY"]
openai_endpoint = st.secrets["openai"]["AZURE_OPENAI_ENDPOINT"]
openai_version = st.secrets["openai"]["AZURE_OPENAI_API_VERSION"]
openai_deployment_name = st.secrets["openai"]["AZURE_OPENAI_DEPLOYMENT_NAME"]


AGENT_INSTRUCTIONS = """You are an assistant for people approving loans to small businesses. Your name, if asked, is 'FRA'.

Wait for specific instructions from the user before taking any action. Do not perform tasks unless they are explicitly requested.
Briefly introduce yourself and explain your purpose when the user first interacts with you.

You may be asked to:
- Use StructureLoanData to structure the loan application in the chat, and use the output for any function that takes a `loan_data` parameter.
- The database contains financial data about the companies, use DatabaseConnector to retrieve it. The output is structured financial data of the company and can be used in any function that takes a `financial_data` parameter.
- Please wtite a narrative about the risk profile of the business, using the financial_data. Explain that you have gathered the financial data from the database. Write a bullet point for each piece of financial data. Include location and sector details. Do not list the thresholds in this narrative. Once you have done this pause to ask for confirmation before proceeding.
- Assess the credit risk profile of an organisation based on model outputs, we are trying to predict the chance of the business of defaulting on the loan. Express the output as a percentage of default.
- Check the survivability of a small business using our model. It predicts the chance of the business surviving for 3 years. Express the output as a percentage of survival.
- Once you have the risk score and survival probability, pause to ask the user if they want to proceed with the loan approval checks.
- Determine whether loan is approvable, you will need survivability score and credit risk score from those models first
- If the loan is approvable, you can then determine the interest rate using the InterestRate plugin. The interest rate is based on the risk score and survival probability.
- Once you have determined if a loan is approvale, you can return the verdict to the user. That finishes the conversation.
- If the loan isn't approvable, you can suggest a counterfactual to the user, which will help them understand what they need to do to get the loan approved. When the user asks for counterfactuals, automaticcally suggest financial data changes to the user, which are listed below.
- Change the values of the financial data with user approval, and then re-run the risk assessment, survivability and loan approval checks.
- Key thresholds for risk evaluation:
  - EBITDA margin: below 10% increases risk, above 20% decreases risk
  - Debt-to-equity ratio: above 2.5 increases risk, below 1.0 decreases risk
  - Current ratio: below 1.0 increases risk, above 1.5 decreases risk
  - Revenue growth: negative increases risk, above 7% decreases risk
- Key thresholds for loan approval:
  - Above 50% risk score or below 50% survival probability results in rejection
  - Loan amount above 100,000 with risk score above 30% results in rejection
  - Only suggesting lowering the amount if the risk score is above 30% and the loan amount is above 100,000


You can try and extract structured data from the chat using the StructureLoanData plugin, which will help you understand the loan application better. With any message

"""

def build_agent(claim_text):
    kernel = Kernel()
    kernel.add_service(AzureChatCompletion(
        deployment_name="kainosgpt",
        endpoint=openai_endpoint,
        api_key=openai_key
    ))



    # 👉 Keep RAG setup for policy lookup
    vector_memory_rag = VectorMemoryRAGPlugin()
    if claim_text:
        vector_memory_rag.add_document(claim_text)

    # --- Register plugins
    kernel.add_plugin(MockDatabaseConnector(), plugin_name="DatabaseConnector")
    kernel.add_plugin(vector_memory_rag, plugin_name="VectorMemoryRAG")
    kernel.add_plugin(RiskEvaluator(), plugin_name="RiskModel")
    kernel.add_plugin(IsLoanApprovable(), plugin_name="IsLoanApprovable")
    kernel.add_plugin(InterestRate(), plugin_name="InterestRate")
    kernel.add_plugin(Counterfactual(), plugin_name="Counterfactual")
    kernel.add_plugin(SurvivabilityEstimator(), plugin_name="SurvivabilityEstimator")
    kernel.add_plugin(StructureLoanData(kernel), plugin_name="StructureLoanData")



    agent = ChatCompletionAgent(
        kernel=kernel,
        name="FRA",
        instructions=AGENT_INSTRUCTIONS,
        arguments=KernelArguments(
            settings=OpenAIChatPromptExecutionSettings(
                temperature=0.5,
                top_p=0.95,
                function_choice_behavior=FunctionChoiceBehavior.Auto()
            )
        )
    )

    return agent

# Keep responses brief—no more than a few paragraphs—and always respond only to what the user has asked, when they ask it. 