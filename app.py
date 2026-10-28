import traceback
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field

# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")


# --------------------------------------------------
# Tool used by Agent 3 to save the support record
# --------------------------------------------------

class SaveRecordInput(BaseModel):
    request_id: str = Field(description="Unique support request ID")
    query: str = Field(description="Original user query")
    answer1: str = Field(description="Answer from Assistant")
    answer2: str = Field(description="Answer from Web Search Assistant")
    sources: str = Field(
        default="No sources supplied",
        description="Web source URLs, if available",
    )


class SaveSupportRecordTool(BaseTool):
    name: str = "save_support_record"
    description: str = (
        "Save the original query and both assistant answers to a "
        "UTF-8 text file in the output directory. Call this tool "
        "after receiving the answers."
    )
    args_schema: type[BaseModel] = SaveRecordInput

    def _run(
        self,
        request_id: str,
        query: str,
        answer1: str,
        answer2: str,
        sources: str = "No sources supplied",
    ) -> str:
        # Restrict the filename to safe characters.
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", request_id)
        path = OUTPUT_DIR / f"support_{safe_id}.txt"

        content = (
            f"Support Request ID: {safe_id}\n"
            f"Timestamp: {datetime.now().astimezone().isoformat()}\n"
            f"\n{'=' * 60}\n"
            f"USER QUERY\n{'=' * 60}\n{query}\n"
            f"\n{'=' * 60}\n"
            f"ANSWER 1 - ASSISTANT\n{'=' * 60}\n{answer1}\n"
            f"\n{'=' * 60}\n"
            f"ANSWER 2 - WEB SEARCH ASSISTANT\n{'=' * 60}\n{answer2}\n"
            f"\n{'=' * 60}\n"
            f"WEB SOURCES\n{'=' * 60}\n{sources}\n"
        )

        path.write_text(content, encoding="utf-8")
        return f"Record saved successfully: {path}"


# --------------------------------------------------
# Three agents and three sequential tasks
# --------------------------------------------------

def build_crew(query: str, request_id: str):
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is missing. Configure it in your .env file."
        )

    if not os.getenv("SERPER_API_KEY"):
        raise ValueError(
            "SERPER_API_KEY is missing. Configure it in your .env file."
        )

    # Agent 1: Direct answer, without web-search tools.
    assistant = Agent(
        role="Assistant",
        goal="Answer the user's query clearly using your existing knowledge.",
        backstory=(
            "You are a helpful customer support assistant. "
            "Answer directly, distinguish facts from uncertainty, "
            "and never pretend you searched the web."
        ),
        llm=MODEL_NAME,
        allow_delegation=False,
        verbose=False,
    )

    # Agent 2: Web-grounded answer.
    web_search_assistant = Agent(
        role="Web Search Assistant",
        goal=(
            "Research the user's query online and provide an accurate, "
            "useful answer supported by web sources."
        ),
        backstory=(
            "You are a careful online researcher. Use search results "
            "to verify current information. Include source titles and "
            "full URLs in your answer. If evidence is insufficient, "
            "say so rather than inventing sources."
        ),
        tools=[SerperDevTool()],
        llm=MODEL_NAME,
        allow_delegation=False,
        verbose=False,
    )

    # Agent 3: Saves the query and both answers.
    entry_agent = Agent(
        role="Entry Agent",
        goal=(
            "Preserve the original query and both preceding answers "
            "in a readable text record, then confirm the save."
        ),
        backstory=(
            "You are a meticulous support-record administrator. "
            "Use the save_support_record tool with the exact query "
            "and both answers. Do not fabricate save confirmations."
        ),
        tools=[SaveSupportRecordTool()],
        llm=MODEL_NAME,
        allow_delegation=False,
        verbose=False,
    )

    task1 = Task(
        description=(
            "Answer this user query from your own knowledge only. "
            "Do not use web search or claim to have checked current "
            "sources.\n\nUser query: {query}"
        ),
        expected_output=(
            "A clear, self-contained direct answer. State uncertainty "
            "where appropriate."
        ),
        agent=assistant,
    )

    task2 = Task(
        description=(
            "Search the web for the following query and answer it "
            "using relevant search results. Include source titles and "
            "full URLs in the answer. Separate verified facts from "
            "uncertainty.\n\nUser query: {query}"
        ),
        expected_output=(
            "A web-grounded answer with relevant source titles and URLs."
        ),
        agent=web_search_assistant,
        context=[task1],
    )

    task3 = Task(
        description=(
            "Save the original query and both preceding answers using "
            "the save_support_record tool. Use these exact values:\n"
            f"Request ID: {request_id}\n"
            f"Original query: {query}\n\n"
            "Answer 1 is the output of the Assistant task.\n"
            "Answer 2 is the output of the Web Search Assistant task.\n"
            "Extract the full answers without replacing them with "
            "summaries. Include any web URLs in the sources field.\n"
            "Call the tool and report its actual result."
        ),
        expected_output=(
            "A confirmation of the file-writing tool's result and "
            "the saved record location."
        ),
        agent=entry_agent,
        context=[task1, task2],
    )

    # Exactly three agents, in the required order.
    crew = Crew(
        agents=[assistant, web_search_assistant, entry_agent],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        memory=True,
        verbose=True,
         output_log_file="logs.txt",
    )

    return crew


# --------------------------------------------------
# Extract source URLs and persist a verified record
# --------------------------------------------------

def extract_urls(text: str) -> list[str]:
    """Extract unique HTTP(S) URLs from an answer."""
    urls = re.findall(r"https?://[^\s<>\"')\]]+", text or "")
    return list(dict.fromkeys(url.rstrip(".,;") for url in urls))


def save_verified_record(
    request_id: str,
    query: str,
    answer1: str,
    answer2: str,
) -> Path:
    """
    Ensure the record exists with the actual task outputs.
    This also repairs an incomplete or missing agent-written file.
    """
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", request_id)
    path = OUTPUT_DIR / f"support_{safe_id}.txt"

    urls = extract_urls(answer2)
    sources = "\n".join(urls) if urls else "No URLs extracted from the answer."

    content = (
        f"Support Request ID: {safe_id}\n"
        f"Timestamp: {datetime.now().astimezone().isoformat()}\n\n"
        f"{'=' * 60}\nUSER QUERY\n{'=' * 60}\n{query}\n\n"
        f"{'=' * 60}\nANSWER 1 - ASSISTANT\n{'=' * 60}\n"
        f"{answer1}\n\n"
        f"{'=' * 60}\nANSWER 2 - WEB SEARCH ASSISTANT\n{'=' * 60}\n"
        f"{answer2}\n\n"
        f"{'=' * 60}\nWEB SOURCES\n{'=' * 60}\n{sources}\n"
    )

    path.write_text(content, encoding="utf-8")

    # Verify that the saved file can be read.
    saved_text = path.read_text(encoding="utf-8")
    if query not in saved_text or answer1 not in saved_text or answer2 not in saved_text:
        raise IOError("Saved record verification failed.")

    return path


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(
    page_title="CrewAI Customer Support",
    page_icon="💬",
    layout="wide",
)

st.title("💬 CrewAI Customer Support System")
st.caption(
    "Three sequential agents · Web research · CrewAI memory · "
    "Verified text records"
)

with st.sidebar:
    st.subheader("Configuration")
    st.write(f"LLM: `{MODEL_NAME}`")
    st.write("CrewAI memory: Enabled")
    st.write("Workflow: Sequential")
    st.write("Agents: 3")
    st.caption("Do not share your API keys or commit your .env file.")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_area(
    "Enter your query or support task",
    placeholder="Example: How can I reset my account password securely?",
    height=120,
    key="support_query",
)

run_button = st.button(
    "Run support workflow",
    type="primary",
    disabled=not query.strip(),
)

if run_button:
    request_id = uuid.uuid4().hex[:12]

    try:
        with st.spinner(
            "Running Assistant → Web Search Assistant → Entry Agent..."
        ):
            crew = build_crew(query.strip(), request_id)

            result = crew.kickoff(
                inputs={"query": query.strip()}
            )


            # Sequential task outputs are used rather than relying
            # only on the final Entry Agent response.
            task_outputs = result.tasks_output
            if len(task_outputs) < 3:
                raise RuntimeError(
        f"Expected 3 task outputs, got {len(task_outputs)}. "
        f"Crew result: {result.raw}"
        )

            answer1 = str(task_outputs[0].raw or "").strip()
            answer2 = str(task_outputs[1].raw or "").strip()
            entry_result = str(task_outputs[2].raw or "").strip()

            # Temporary diagnostics in the CMD terminal.
            print("Task 1 output:", repr(answer1[:300]))
            print("Task 2 output:", repr(answer2[:300]))
            print("Task 3 output:", repr(entry_result[:300]))

            if not answer1:
                answer1 = "The Assistant did not return an answer."
            if not answer2:
                answer2 = "The Web Search Assistant did not return an answer."

            # Verify the record even if the Entry Agent's tool call
            # was skipped or failed.
            record_path = save_verified_record(
                request_id=request_id,
                query=query.strip(),
                answer1=answer1,
                answer2=answer2,
            )

        item = {
            "request_id": request_id,
            "query": query.strip(),
            "answer1": answer1,
            "answer2": answer2,
            "entry_result": entry_result,
            "record_path": str(record_path),
            "record_text": record_path.read_text(encoding="utf-8"),
            "timestamp": datetime.now().astimezone().isoformat(),
        }

        st.session_state.history.insert(0, item)
        st.session_state.current_request = request_id
        st.success(f"Workflow completed. Record saved: {record_path}")

    except Exception as exc:
        st.error("The support workflow failed.")
        st.write(f"Error type: {type(exc).__name__}")
        st.code(str(exc))

        # Full diagnostic details in the CMD terminal.
        traceback.print_exc()
        st.error(
            "The workflow could not be completed. Check your API "
            "configuration, network access, installed package versions, "
            "and terminal logs for details."
        )
        st.caption(f"Error type: {type(exc).__name__}")

# Show the latest result.
current_id = st.session_state.get("current_request")
current = next(
    (
        item for item in st.session_state.history
        if item["request_id"] == current_id
    ),
    None,
)

if current:
    st.divider()
    st.subheader("Support results")
    st.caption(
        f"Request ID: {current['request_id']} · "
        f"{current['timestamp']}"
    )

    with st.expander("Original query", expanded=True):
        st.write(current["query"])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 1. Assistant")
        st.markdown(current["answer1"])

    with col2:
        st.markdown("### 2. Web Search Assistant")
        st.markdown(current["answer2"])

        urls = extract_urls(current["answer2"])
        if urls:
            st.markdown("**Extracted source links**")
            for url in urls:
                st.markdown(f"- [{url}]({url})")

    with st.expander("3. Entry Agent — save result"):
        st.write(current["entry_result"])
        st.code(current["record_path"], language="text")

    st.download_button(
        label="Download support record (.txt)",
        data=current["record_text"],
        file_name=f"support_{current['request_id']}.txt",
        mime="text/plain",
    )

if st.session_state.history:
    st.divider()
    st.subheader("Session history")

    for item in st.session_state.history:
        label = f"{item['timestamp']} — {item['query'][:70]}"
        with st.expander(label):
            st.markdown("**Assistant answer**")
            st.write(item["answer1"])
            st.markdown("**Web Search answer**")
            st.write(item["answer2"])
            st.caption(f"Record: {item['record_path']}")