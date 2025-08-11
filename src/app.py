import streamlit as st
import asyncio
from io import StringIO
import contextlib
import json
import nest_asyncio

from main import main as run_main


# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

import os
knos_path = os.path.join(os.path.dirname(__file__), "../img/knos.png")
screenshot_path = os.path.join(os.path.dirname(__file__), "../img/Screenshot 2025-06-30 135757.png")

def get_message_content(message) -> str:
    if not message.content:
        return ""
    return str(message.content)

# Page config
st.set_page_config(page_title="Kainos Underwriting Assistant", layout="wide")
st.title("Kainos Agentic Underwriting Assistant")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_thread" not in st.session_state:
    st.session_state.agent_thread = None
if "metrics" not in st.session_state:
    st.session_state.metrics = {"total_tokens": 0, "total_steps": 0}
if "claim_text" not in st.session_state:
    st.session_state.claim_text = None
if "output" not in st.session_state:
    st.session_state.output = ""

# Sidebar
st.sidebar.image(knos_path, width=200)
st.sidebar.header("🔧 How can I help?")
st.sidebar.text("I'm an agentic insurance underwriting assistant; able to support you with analysing businesses insurance profiles, providing premium estimates, assessing a company's risk profile and more")


# --- Clear history ---
st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.messages = []
    st.session_state.agent_thread = None
    st.session_state.metrics = {"total_tokens": 0, "total_steps": 0}
    st.session_state.document_appended = None
    st.session_state.claim_text = None   
    uploaded_file = None     
    st.success("Chat history and uploaded memory cleared")

# --- Display the contents of claim_text for debugging ---
if st.session_state.get("claim_text"):
    st.sidebar.markdown("#### Uploaded Claim Document Content:")
    st.sidebar.code(st.session_state.claim_text, language="text")

# --- Chat Input ---
user_input = st.chat_input("Type your message and press Enter...")

# --- Append document to user input only the first time after upload ---
if (
    user_input is not None and user_input.strip()
    and st.session_state.get("claim_text")
    and not st.session_state.get("document_appended", False)
):
    user_input = (
        f"{user_input}\n\n"
        "-----\n"
        "Uploaded Document Contents:\n"
        "-----\n"
        "```text\n"
        f"{st.session_state.claim_text.rstrip()}\n\n"
        "-----\n"# Ensures the last line gets its own newline
        "```\n"
    )
    st.session_state.document_appended = True

async def handle_user_input(user_input):
    buffer = StringIO()
    with contextlib.redirect_stdout(buffer):
        response = await run_main(
            user_input,
            st.session_state.agent_thread,
            st.session_state.claim_text 
        )
    output = buffer.getvalue()
    if response:
        st.session_state.messages.extend(response.messages)
        st.session_state.agent_thread = response.thread
        st.session_state.metrics["total_tokens"] = response.metrics.get("total_tokens", 0)
        st.session_state.metrics["total_steps"] = response.metrics.get("steps", 0)
    return output

def run_async(coroutine):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)

if user_input is not None and user_input.strip():
    # Use the run_async helper function
    st.session_state.output = run_async(handle_user_input(user_input))

# --- Tabs ---
tab_main, tab_diag, tab_hiw = st.tabs(["💬 Main", "🛠 Diagnostics", "How it works"])

with tab_main:
    if user_input is not None and user_input.strip():
        for message in st.session_state.messages:
            content = get_message_content(message)
            if content:
                with st.chat_message(message.role.lower()):
                    name = message.name or message.role
                    st.markdown(f"**{name}:** {content}")
    else:
        st.warning("Please enter a prompt.")

with tab_diag:
    st.markdown("## Message Timeline")
    if st.session_state.messages:
        timeline_steps = []
        for message in st.session_state.messages:
            role = message.role.lower()
            if hasattr(message, 'function_call') and message.function_call:
                timeline_steps.append("⚙️ Function")
            elif hasattr(message, 'function_response') and message.function_response:
                timeline_steps.append("📤 Result")
            elif role == "assistant":
                timeline_steps.append("🤖 Assistant")
            elif role == "user":
                timeline_steps.append("🧑 User")
            elif role == "system":
                timeline_steps.append("⚡ System")
            else:
                timeline_steps.append("💬 " + role.title())
        st.markdown(" ➔ ".join(timeline_steps))

    st.markdown("## Message Trace")
    for i, message in enumerate(st.session_state.messages):
        with st.expander(f"Step {i + 1}: {message.role.title()}", expanded=False):
            content = get_message_content(message)
            if content:
                st.markdown("**🧠 Content:**")
                st.code(content, language="markdown")
            if message.function_call:
                st.markdown(f"**🤖 Function Call:** `{message.function_call['name']}`")
                st.markdown("**📥 Arguments:**")
                st.code(json.dumps(message.function_call["arguments"], indent=2), language="json")
            if message.function_response:
                st.markdown("**🛠 Function Result:**")
                st.code(str(message.function_response), language="markdown")
            if message.metadata:
                st.markdown("**📊 Metadata:**")
                st.json(message.metadata)

    st.markdown("## Metrics")
    col1, col2 = st.columns(2)
    with col1:
        current_total = st.session_state.metrics.get("total_tokens", 0)
        st.metric("Total Tokens", current_total)
    with col2:
        total_steps = len(st.session_state.messages)
        st.metric("Total Steps", total_steps)

    with st.expander("💰 Token Usage & Cost", expanded=False):
        total_prompt_tokens = 0
        total_completion_tokens = 0
        for message in st.session_state.messages:
            if message.metadata and "usage" in message.metadata:
                usage = message.metadata["usage"]
                total_prompt_tokens += usage.prompt_tokens
                total_completion_tokens += usage.completion_tokens
        if total_prompt_tokens > 0 or total_completion_tokens > 0:
            total_tokens = total_prompt_tokens + total_completion_tokens
            PROMPT_COST_PER_1M = 0.1
            COMPLETION_COST_PER_1M = 0.4
            prompt_cost = (total_prompt_tokens * PROMPT_COST_PER_1M) / 1_000_000
            completion_cost = (total_completion_tokens * COMPLETION_COST_PER_1M) / 1_000_000
            total_cost = prompt_cost + completion_cost
            st.markdown(f"""
            ### Current Request
            - Prompt Tokens: `{total_prompt_tokens}`
            - Completion Tokens: `{total_completion_tokens}`
            - Total Tokens: `{total_tokens}`
            
            ### Cost Breakdown
            - Prompt Cost `${prompt_cost:.4f}`
            - Completion Cost `${completion_cost:.4f}`
            - **Total Cost: `${total_cost:.4f}`**
            """)
        else:
            st.info("No usage data available for current request")

    with st.expander("🖥️ Console Output", expanded=False):
        if st.session_state.output and st.session_state.output.strip():
            st.code(st.session_state.output, language="bash")
        else:
            st.info("No console output captured")

with tab_hiw:
    static_container = st.container()
    
with static_container:
    st.info("This is a live demo — feel free to try asking a question!")
    st.header("🤖 Powered by Agentic AI and Microsoft Semantic Kernel")

    with st.expander("🧠 What This Demo Does", expanded=True):
        st.markdown("""
        This demo showcases a conversational underwriter assistant built with **Microsoft Semantic Kernel**, **Amazon Bedrock**, **Amazon Dynamo DB** and **Amazon SageMaker**.

        It acts like a professional insurance consultant, combining memory, functions, and decision-making tools to help process claims and customer questions — all in real-time.
        """)

    with st.expander("🚀 Key Capabilities"):
        st.markdown("""
        - 🔍 **Document Understanding**  
        Upload policy documents or claim descriptions — the assistant understands and stores them for later use.

        - 🧠 **Long-Term Memory with Semantic Search**  
        The assistant stores claims and documents in collections and can retrieve the most relevant ones when needed, just like human memory.

        - 🏷️ **Model Calling**  
        Able to call external models such as risk scoring or fraud detection tools.

        - ✅ **Augmented Decision-Making**  
        Approves, rejects, or refers claims using embedded decision logic.

        - ⚖️ **Regulatory Compliance Checks**  
        Evaluates decision language against **UK Consumer Duty** principles.
        """)

    with st.expander("🧩 How Agentic AI Works Here"):
        st.markdown("""
        - The assistant is built as an **agent** — it has memory, goals, tools, and autonomy.
        - It uses **Semantic Kernel Plugins** to run tools (like risk, compliance, estimation).
        - It retains memory across interactions — giving it context and recall.
        - It autonomously decides **when** to use tools and **how** to respond.
        """)
        st.image(screenshot_path, caption="System Architecture Overview")

    with st.expander("💡 Why Use Semantic Kernel?"):
        st.markdown("""
        Compared to Copilot Studio or low-code tools like Power Automate, Semantic Kernel offers:
        
        - 🔄 **Model Flexibility**  
        Bring your own LLMs (OpenAI, Azure OpenAI, Hugging Face, etc.)

        - 🧩 **Plugin Modularity**  
        Add/remove plugins (e.g., risk model, compliance check) without rewriting flows.

        - 🧠 **Real Memory**  
        Semantic recall across multiple conversations/documents.

        - 🧾 **Full Traceability**  
        Every decision can be traced to its input, tools, and reasoning.

        - 🛠️ **Code-Driven, Not Box-Driven**  
        Avoid no-code constraints — every component is inspectable and extensible.
        
        - 🖥️ **MCP and A2A compatible**  
        Easily connect to Model Context Protocol (MCP) servers to add new tools or A2A to interact with other agents.

        - 💰 **Transparent Pricing**  
        No per-flow licensing. You control usage via LLM APIs.
        """)
        
       

    st.markdown("---")
    st.caption("Built with 🧠 Semantic Kernel, 🗂️ Streamlit, and ⚖️ OpenAI")
