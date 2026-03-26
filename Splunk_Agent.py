# splunk_agent.py
# Natural-language -> SPL -> run SPL via Splunk REST -> summarize results via LLM

import os
import re
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv
import ollama

# Load environment variables
load_dotenv()


FORBIDDEN_SPL_KEYWORDS = [
    "outputlookup",
    "| outputlookup",
    "collect",
    "dbxquery",
    "sendemail",
    "restart",
    "delete",
    "update",
    "insert",
    "| map",
    "script",
]


def _assert_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _extract_spl(text: str) -> str:
    """
    Extract SPL from common LLM formats like:
      - ```spl ... ```
      - ``` ... ```
      - plain text query
    """
    # Prefer fenced block
    m = re.search(r"```(?:spl)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Otherwise take first non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    return lines[0] if lines else text.strip()


def _reject_forbidden_spl(spl: str) -> None:
    lowered = spl.lower()
    for kw in FORBIDDEN_SPL_KEYWORDS:
        if kw in lowered:
            raise ValueError(f"Refusing to run forbidden SPL keyword: {kw}")


class SplunkClient:
    def __init__(
        self,
        base_url: str,
        verify_ssl: bool = True,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        timeout_s: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.verify_ssl = verify_ssl
        self.timeout_s = timeout_s

        self.session = requests.Session()
        if token:
            # Some Splunk setups use bearer tokens; adjust if yours differs.
            self.session.headers.update({"Authorization": f"Bearer {token}"})
        else:
            if username is None or password is None:
                raise RuntimeError("Provide either token OR username+password for Splunk auth.")
            self.session.auth = (username, password)

    def _create_search_job(
        self,
        search_spl: str,
        earliest: str = "-24h",
        latest: str = "now",
        max_count: int = 50,
    ) -> str:
        """
        Creates a search job and returns the `sid`.
        """
        url = f"{self.base_url}/services/search/jobs"
        data = {
            "search": search_spl,
            "earliest_time": earliest,
            "latest_time": latest,
            "output_mode": "json",
            "max_count": str(max_count),
        }

        resp = self.session.post(url, data=data, verify=self.verify_ssl, timeout=self.timeout_s)
        resp.raise_for_status()

        # Response can be JSON or XML depending on Splunk config.
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "application/json" in ctype:
            j = resp.json()
            # Common: {"sid":"..."} or {"entry":[{"id":"...","name":"..."}]}
            if "sid" in j:
                return j["sid"]
            # fallback: look for an sid-like value in JSON
            for k, v in j.items():
                if isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9-]{8,}", v):
                    return v

        # XML fallback: <sid>...</sid>
        m = re.search(r"<sid>(.*?)</sid>", resp.text, flags=re.DOTALL)
        if not m:
            raise RuntimeError(f"Could not extract sid from Splunk response: {resp.text[:300]}")
        return m.group(1).strip()

    def _wait_for_job(self, sid: str, poll_s: float = 1.0, timeout_s: int = 60) -> None:
        url = f"{self.base_url}/services/search/jobs/{sid}"
        start = time.time()
        while True:
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timed out waiting for Splunk job {sid}")

            params = {"output_mode": "json"}
            resp = self.session.get(url, params=params, verify=self.verify_ssl, timeout=self.timeout_s)
            resp.raise_for_status()
            j = resp.json()

            # Typical: {"entry":[{"content":{"dispatchState":"DONE","isDone":"1"}}]}
            entries = j.get("entry") or []
            if entries:
                content = entries[0].get("content") or {}
                dispatch_state = content.get("dispatchState")
                is_done = str(content.get("isDone", "")).lower() in {"1", "true", "yes"}
                if dispatch_state == "DONE" or is_done:
                    return
                if dispatch_state == "FAILED":
                    raise RuntimeError(f"Splunk search job failed (sid={sid}): {j}")
            time.sleep(poll_s)

    def _get_results(
        self,
        sid: str,
        max_results: int = 20,
    ) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/services/search/jobs/{sid}/results"
        params = {"output_mode": "json", "count": str(max_results)}
        resp = self.session.get(url, params=params, verify=self.verify_ssl, timeout=self.timeout_s)
        resp.raise_for_status()
        j = resp.json()

        # Typical: {"results":[{...},{...}]}
        results = j.get("results")
        if isinstance(results, list):
            return results
        return []

    def search(
        self,
        search_spl: str,
        earliest: str = "-24h",
        latest: str = "now",
        max_results: int = 20,
    ) -> Dict[str, Any]:
        # max_count influences job output; results endpoint uses `count`.
        sid = self._create_search_job(
            search_spl=search_spl,
            earliest=earliest,
            latest=latest,
            max_count=max_results,
        )
        self._wait_for_job(sid=sid)
        rows = self._get_results(sid=sid, max_results=max_results)

        return {
            "sid": sid,
            "earliest": earliest,
            "latest": latest,
            "max_results": max_results,
            "results": rows,
        }


def call_llm_ollama(prompt: str, model: str, temperature: float = 0.0) -> str:
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": temperature,
            },
        )
        return response['message']['content'].strip()
    except Exception as e:
        raise RuntimeError(f"Ollama API error: {str(e)}")


def generate_spl(
    question: str,
    openai_model: str,
    splunk_indexes_hint: Optional[str] = None,
) -> str:
    splunk_indexes_hint = splunk_indexes_hint or "Any relevant index/sourcetype."

    prompt = f"""
You translate natural-language questions into Splunk SPL.
Rules:
- Output ONLY a single SPL query (no explanation).
- Prefer using the Search command and pipelines.
- Always include a reasonable limit (e.g., `| head 20`) to keep results small.
- Use only read-only operations.
- The question: {question}
- Index/Sourcetype hint: {splunk_indexes_hint}

Return SPL:
"""
    out = call_llm_ollama(prompt=prompt, model=openai_model, temperature=0.0)
    spl = _extract_spl(out)
    if not spl:
        raise RuntimeError("LLM did not return an SPL query.")
    return spl


def summarize_results(
    question: str,
    spl_query: str,
    splunk_search_response: Dict[str, Any],
    openai_model: str,
) -> str:
    # Keep payload small for the model
    results = splunk_search_response.get("results", [])[:20]
    payload = json.dumps(results, ensure_ascii=False)

    prompt = f"""
You are answering a user question using Splunk search results.

User question:
{question}

Used SPL:
{spl_query}

Search results (JSON array, may be empty):
{payload}

Write a concise answer. If results are empty, say that clearly and suggest what to check (e.g., time range, index).
"""
    return call_llm_ollama(prompt=prompt, model=openai_model, temperature=0.2)


def run_agent(
    question: str,
    earliest: str,
    latest: str,
    max_results: int,
    splunk_indexes_hint: Optional[str],
    openai_model: str,
    splunk_base_url: Optional[str] = None,
    splunk_token: Optional[str] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    splunk_base_url = "https://redcube.qualcomm.com:8000/"
    #splunk_base_url = _assert_env("SPLUNK_BASE_URL")  # e.g. https://splunk:8089
    #verify_ssl = os.getenv("SPLUNK_VERIFY_SSL", "true").strip().lower() in {"1", "true", "yes"}
    #splunk_username = os.getenv("SPLUNK_USERNAME")
    #splunk_password = os.getenv("SPLUNK_PASSWORD")
    splunk_token = os.getenv("SPLUNK_TOKEN")
   

    spl = generate_spl(
        question=question,
        openai_model=openai_model,
        splunk_indexes_hint=splunk_indexes_hint,
    )
    _reject_forbidden_spl(spl)

    sc = SplunkClient(
        base_url=splunk_base_url,
        verify_ssl=verify_ssl,
        username=splunk_username,
        password=splunk_password,
        token=splunk_token,
    )

    resp = sc.search(
        search_spl=spl,
        earliest=earliest,
        latest=latest,
        max_results=max_results,
    )

    answer = summarize_results(
        question=question,
        spl_query=spl,
        splunk_search_response=resp,
        openai_model=openai_model,
    )

    return answer, spl, resp


def main():
    # Streamlit page configuration
    st.set_page_config(
        page_title="Splunk Agent",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("🔍 Splunk Natural Language Agent")
    st.markdown("Convert natural language questions into Splunk queries and get AI-powered insights.")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("🔗 Splunk Connection")
        splunk_base_url = st.text_input(
            "Splunk Base URL",
            value="https://localhost:8089/",
            help="e.g., https://splunk.example.com:8089/ or http://localhost:8089/"
        )
        splunk_token = st.text_input(
            "Splunk Auth Token",
            type="password",
            value="",
            help="Bearer token for Splunk authentication"
        )
        
        st.divider()
        st.subheader("🔍 Search Options")
        earliest = st.text_input(
            "Earliest time",
            value="-24h",
            help="Time range for Splunk search (e.g., -24h, -7d, -1h)"
        )
        latest = st.text_input(
            "Latest time",
            value="now",
            help="End time for Splunk search"
        )
        max_results = st.slider(
            "Max results",
            min_value=5,
            max_value=100,
            value=20,
            step=5
        )
        splunk_indexes_hint = st.text_input(
            "Index/Sourcetype hint",
            value="",
            placeholder="Leave empty for any index",
            help="Hint to the LLM about which Splunk index to search"
        )
        
        st.divider()
        st.subheader("🤖 LLM Options")
        openai_model = st.selectbox(
            "Ollama Model",
            options=["llama2", "llama3", "llama3.1", "mistral", "neural-chat", "starling"],
            index=0
        )
        
        st.divider()
        st.markdown("### 📋 Information")
        st.info(
            "This agent translates natural language questions into SPL queries, "
            "executes them against your Splunk instance, and summarizes the results using Ollama."
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        question = st.text_area(
            "Ask a question about your Splunk data:",
            height=100,
            placeholder="e.g., 'How many errors occurred in the last 24 hours?'"
        )
    
    with col2:
        st.markdown("### Actions")
        submit_button = st.button("🚀 Execute Query", use_container_width=True, type="primary")
    
    # Execute query
    if submit_button and question:
        if not splunk_base_url or not splunk_token:
            st.error("❌ Please enter Splunk URL and Auth Token in the configuration panel.")
            return
        
        try:
            with st.spinner("🔄 Processing your question..."):
                answer, spl, search_response = run_agent(
                    question=question,
                    earliest=earliest,
                    latest=latest,
                    max_results=max_results,
                    splunk_indexes_hint=splunk_indexes_hint or None,
                    openai_model=openai_model,
                    splunk_base_url=splunk_base_url,
                    splunk_token=splunk_token,
                )
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["📝 Answer", "🔤 SPL Query", "📊 Raw Results"])
            
            with tab1:
                st.markdown("### AI-Powered Answer")
                st.markdown(answer)
            
            with tab2:
                st.markdown("### Generated SPL Query")
                st.code(spl, language="sql")
                st.caption("This is the Splunk query that was executed.")
            
            with tab3:
                st.markdown("### Raw Search Results")
                results = search_response.get("results", [])
                st.json(results[:10])  # Show first 10 results
                if len(results) > 10:
                    st.info(f"Showing 10 of {len(results)} results")
            
            # Additional info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Search ID", search_response.get("sid", "N/A")[:12])
            with col2:
                st.metric("Results Count", len(results))
            with col3:
                st.metric("Time Range", f"{earliest} to {latest}")
            with col4:
                st.metric("Ollama Model", openai_model)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    
    elif submit_button:
        st.warning("⚠️ Please enter a question first.")


if __name__ == "__main__":
    main()
