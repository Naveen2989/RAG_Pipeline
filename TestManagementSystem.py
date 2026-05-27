#!/usr/bin/env python3
"""
multiagent_test_system.py

Single-file Multi-Agent Test Management & Requirement Ingestion System.

Capabilities:
 - Accept multiple requirement files (.txt, .docx, .pdf, .png/.jpg)
 - Extract text (PyPDF2, python-docx, pytesseract for images)
 - Multi-agent pipelines using autogen + local Ollama model:
      requirements_agent -> test_planner_agent -> testcase_agent -> testdata_agent -> integrator_agent
 - Integrator executes JIRA/Zephyr REST calls if configured via env
 - CLI usage: python multiagent_test_system.py --files requirements.docx notes.pdf screenshot1.png

Requirements (install):
 pip install autogen python-dotenv python-docx PyPDF2 pillow pytesseract requests

For OCR (images), install Tesseract separately and ensure 'tesseract' is in PATH.
Place configuration in 'api.env' file (or environment variables):
 - For Ollama local model: nothing required aside from running Ollama & model available
 - Optional JIRA / Zephyr:
    JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY
    ZEPHYR_BASE_URL, ZEPHYR_API_TOKEN

Author: Generated for a software test engineer
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from dotenv import load_dotenv
import requests

# File parsing imports
from PyPDF2 import PdfReader
import docx
from PIL import Image
import pytesseract

# AutoGen imports (Microsoft AutoGen)
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Load environment variables from api.env (if present)
load_dotenv("api.env")

# -----------------------
# LLM (Ollama) CONFIG
# -----------------------
# Replace model name with the exact model you have in Ollama (use `ollama list`).
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama:latest")

config_list = [
    {
        "model": OLLAMA_MODEL,
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "api_key": os.getenv("OLLAMA_API_KEY", "ollama"),
        # Add price to silence warnings (local models have no cost)
        "price": [0.0, 0.0],
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.0")),
    "timeout": int(os.getenv("LLM_TIMEOUT", "120")),  # seconds
}

# -----------------------
# Data classes
# -----------------------
@dataclass
class Requirement:
    id: str
    title: str
    type: str
    description: str

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class TestPlan:
    id: str
    title: str
    scope: str
    approach: str
    entry_criteria: List[str]
    exit_criteria: List[str]

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class TestCase:
    id: str
    title: str
    requirement_id: str
    preconditions: List[str]
    steps: List[str]
    expected_result: str
    priority: str = "Medium"

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class TestData:
    testcase_id: str
    data: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)

# -----------------------
# File ingestion utilities
# -----------------------
def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        text = p.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)

def extract_text_from_image(path: str) -> str:
    img = Image.open(path)
    # Optional: convert to grayscale to improve OCR
    try:
        gray = img.convert("L")
    except Exception:
        gray = img
    return pytesseract.image_to_string(gray)

def extract_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return extract_text_from_txt(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    elif ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        return extract_text_from_image(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def load_requirements_from_files(paths: List[str]) -> str:
    texts = []
    for p in paths:
        if not os.path.exists(p):
            texts.append(f"[Error: file not found: {p}]")
            continue
        try:
            text = extract_text_from_file(p)
            header = f"\n--- Extracted from {os.path.basename(p)} ({datetime.fromtimestamp(os.path.getmtime(p)).isoformat()}) ---\n"
            texts.append(header + (text.strip() or "[No text found]"))
        except Exception as e:
            texts.append(f"[Error reading {p}: {repr(e)}]")
    return "\n\n".join(texts)

# -----------------------
# Helper utilities
# -----------------------
def generate_id(prefix: str) -> str:
    return f"{prefix.upper()}-{int(time.time()*1000) % 10000000}"

# -----------------------
# Integration tools (JIRA / Zephyr)
# -----------------------
class IntegrationTools:
    JIRA_BASE = os.getenv("JIRA_BASE_URL")
    JIRA_EMAIL = os.getenv("JIRA_EMAIL")
    JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
    JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY", "TEST")

    ZEPHYR_BASE = os.getenv("ZEPHYR_BASE_URL")
    ZEPHYR_API_TOKEN = os.getenv("ZEPHYR_API_TOKEN")

    @staticmethod
    def _jira_auth():
        if IntegrationTools.JIRA_EMAIL and IntegrationTools.JIRA_API_TOKEN:
            return (IntegrationTools.JIRA_EMAIL, IntegrationTools.JIRA_API_TOKEN)
        return None

    @staticmethod
    def create_jira_issue(title: str, description: str, issuetype: str = "Task") -> dict:
        if not IntegrationTools.JIRA_BASE or not IntegrationTools._jira_auth():
            return {"error": "JIRA credentials not configured"}
        url = f"{IntegrationTools.JIRA_BASE}/rest/api/2/issue"
        payload = {
            "fields": {
                "project": {"key": IntegrationTools.JIRA_PROJECT_KEY},
                "summary": title,
                "description": description,
                "issuetype": {"name": issuetype},
            }
        }
        auth = IntegrationTools._jira_auth()
        headers = {"Content-Type": "application/json"}
        resp = requests.post(url, auth=auth, headers=headers, json=payload, timeout=30)
        if resp.status_code in (200, 201):
            return resp.json()
        else:
            return {"error": f"JIRA API returned {resp.status_code}", "detail": resp.text}

    @staticmethod
    def create_zephyr_testcase(testcase: TestCase) -> dict:
        if not IntegrationTools.ZEPHYR_BASE or not IntegrationTools.ZEPHYR_API_TOKEN:
            return {"error": "Zephyr credentials not configured"}
        # NOTE: Zephyr endpoint varies by product; adjust accordingly
        url = f"{IntegrationTools.ZEPHYR_BASE}/rest/zapi/latest/testcase"
        payload = {
            "name": testcase.title,
            "description": "\n".join(testcase.steps),
            "priority": testcase.priority,
            "preconditions": "\n".join(testcase.preconditions),
            "expectedResult": testcase.expected_result,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {IntegrationTools.ZEPHYR_API_TOKEN}",
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code in (200, 201):
            return resp.json()
        else:
            return {"error": f"Zephyr API returned {resp.status_code}", "detail": resp.text}

# -----------------------
# Multi-agent Test Engineering System
# -----------------------
class TestEngineeringSystem:
    def __init__(self, llm_config: dict):
        self.llm_config = llm_config
        self.setup_agents()
        self.setup_groupchat()

    def setup_agents(self):
        # The UserProxyAgent acts as the orchestrator and executor for tool actions.
        self.user_proxy = UserProxyAgent(
            name="admin",
            human_input_mode="NEVER",
            code_execution_config={"work_dir": "test_output", "use_docker": False},
            system_message="Admin who coordinates test engineering tasks and executes integration actions.",
        )

        # Specialist agents with clear system prompts.
        self.requirements_agent = AssistantAgent(
            name="requirements_agent",
            llm_config=self.llm_config,
            system_message=(
                "You are a Requirements Analyst. Extract and structure business & functional requirements. "
                "Return JSON: {\"requirements\": [{\"id\":\"REQ-1\",\"title\":\"...\",\"type\":\"Business|Functional\",\"description\":\"...\"}, ...]}"
            ),
        )

        self.test_planner_agent = AssistantAgent(
            name="test_planner_agent",
            llm_config=self.llm_config,
            system_message=(
                "You are a Test Planner. Create a Test Plan covering scope, approach, entry_criteria, exit_criteria. "
                "Return JSON with keys id, title, scope, approach, entry_criteria(list), exit_criteria(list)."
            ),
        )

        self.testcase_agent = AssistantAgent(
            name="testcase_agent",
            llm_config=self.llm_config,
            system_message=(
                "You are a Test Case Generator. For given requirements, produce JSON: {\"testcases\":[{id,title,requirement_id,preconditions[],steps[],expected_result,priority},...]}"
            ),
        )

        self.testdata_agent = AssistantAgent(
            name="testdata_agent",
            llm_config=self.llm_config,
            system_message=(
                "You are a Test Data Generator. For each testcase produce JSON: {\"testdata\":[{\"testcase_id\":\"TC-...\",\"data\":{...}}, ...]}"
            ),
        )

        self.integrator_agent = AssistantAgent(
            name="integrator_agent",
            llm_config=self.llm_config,
            system_message=(
                "You are an Integrator. When provided artifacts (testplan/testcases/testdata), produce a JSON plan of actions: "
                "{\"actions\":[{\"type\":\"create_jira\",\"title\":\"...\",\"description\":\"...\"},{\"type\":\"create_zephyr\",\"testcase\":{...}}]} "
                "Do NOT execute APIs; only produce the plan. The UserProxy will execute."
            ),
        )

    def setup_groupchat(self):
        self.agents = [
            self.user_proxy,
            self.requirements_agent,
            self.test_planner_agent,
            self.testcase_agent,
            self.testdata_agent,
            self.integrator_agent,
        ]
        self.group_chat = GroupChat(agents=self.agents, messages=[], max_round=int(os.getenv("GROUPCHAT_MAX_ROUND", "20")))
        self.manager = GroupChatManager(groupchat=self.group_chat, llm_config=self.llm_config)

    # Pipeline methods
    def analyze_requirements(self, raw_requirements_text: str) -> List[Requirement]:
        prompt = f"Extract structured requirements from the text below. Return only JSON.\n\n{raw_requirements_text}"
        resp = self.user_proxy.initiate_chat(self.manager, message=prompt, summary_method="last_msg")
        reqs: List[Requirement] = []
        try:
            parsed = json.loads(resp)
            for r in parsed.get("requirements", []):
                rid = r.get("id") or generate_id("REQ")
                reqs.append(Requirement(
                    id=rid,
                    title=r.get("title", "")[:140],
                    type=r.get("type", "Functional"),
                    description=r.get("description", ""),
                ))
        except Exception:
            # fallback: treat entire text as a single requirement
            reqs.append(Requirement(id=generate_id("REQ"), title="Ingested Requirements", type="Functional", description=resp))
        return reqs

    def create_test_plan(self, requirements: List[Requirement]) -> TestPlan:
        payload = {"requirements": [r.to_dict() for r in requirements]}
        resp = self.user_proxy.initiate_chat(self.manager, message=f"Create a Test Plan JSON from these requirements:\n{json.dumps(payload)}", summary_method="last_msg")
        try:
            parsed = json.loads(resp)
            tp = TestPlan(
                id=parsed.get("id") or generate_id("TPLAN"),
                title=parsed.get("title", "Test Plan"),
                scope=parsed.get("scope", "Full system"),
                approach=parsed.get("approach", "Manual + Automated"),
                entry_criteria=parsed.get("entry_criteria", []),
                exit_criteria=parsed.get("exit_criteria", []),
            )
        except Exception:
            tp = TestPlan(id=generate_id("TPLAN"), title="Generated Test Plan", scope="Full", approach=resp, entry_criteria=[], exit_criteria=[])
        return tp

    def generate_testcases(self, requirements: List[Requirement]) -> List[TestCase]:
        payload = {"requirements": [r.to_dict() for r in requirements]}
        resp = self.user_proxy.initiate_chat(self.manager, message=f"Generate testcases JSON for these requirements:\n{json.dumps(payload)}", summary_method="last_msg")
        tcs: List[TestCase] = []
        try:
            parsed = json.loads(resp)
            for tc in parsed.get("testcases", []):
                tid = tc.get("id") or generate_id("TC")
                tcs.append(TestCase(
                    id=tid,
                    title=tc.get("title", "")[:200],
                    requirement_id=tc.get("requirement_id") or (requirements[0].id if requirements else ""),
                    preconditions=tc.get("preconditions", []),
                    steps=tc.get("steps", []),
                    expected_result=tc.get("expected_result", ""),
                    priority=tc.get("priority", "Medium"),
                ))
        except Exception:
            # fallback: produce one simple test case per requirement
            for r in requirements:
                tcs.append(TestCase(
                    id=generate_id("TC"),
                    title=f"Verify {r.title}",
                    requirement_id=r.id,
                    preconditions=[],
                    steps=[f"Validate requirement {r.id}"],
                    expected_result="Requirement satisfied",
                ))
        return tcs

    def generate_testdata(self, testcases: List[TestCase]) -> List[TestData]:
        payload = {"testcases": [tc.to_dict() for tc in testcases]}
        resp = self.user_proxy.initiate_chat(self.manager, message=f"Generate test data JSON for these testcases:\n{json.dumps(payload)}", summary_method="last_msg")
        results: List[TestData] = []
        try:
            parsed = json.loads(resp)
            for td in parsed.get("testdata", []):
                results.append(TestData(testcase_id=td.get("testcase_id"), data=td.get("data", {})))
        except Exception:
            for tc in testcases:
                results.append(TestData(testcase_id=tc.id, data={"sample_field": "sample_value"}))
        return results

    def integrate_with_jira_and_zephyr(self, testplan: TestPlan, testcases: List[TestCase], testdata: List[TestData]) -> Dict[str, Any]:
        payload = {
            "testplan": testplan.to_dict(),
            "testcases": [tc.to_dict() for tc in testcases],
            "testdata": [td.to_dict() for td in testdata],
        }
        resp = self.user_proxy.initiate_chat(self.manager, message=f"Prepare integration actions JSON for these artifacts:\n{json.dumps(payload)}", summary_method="last_msg")
        executed = {"jira": [], "zephyr": [], "errors": []}
        try:
            plan = json.loads(resp)
            actions = plan.get("actions", [])
            for act in actions:
                if act.get("type") == "create_jira":
                    title = act.get("title")
                    desc = act.get("description", "")
                    issue_type = act.get("issuetype", "Task")
                    res = IntegrationTools.create_jira_issue(title=title, description=desc, issuetype=issue_type)
                    executed["jira"].append(res)
                elif act.get("type") == "create_zephyr":
                    tc_obj = act.get("testcase")
                    # simple mapping into TestCase dataclass
                    try:
                        tc = TestCase(
                            id=tc_obj.get("id") or generate_id("TC"),
                            title=tc_obj.get("title"),
                            requirement_id=tc_obj.get("requirement_id", ""),
                            preconditions=tc_obj.get("preconditions", []),
                            steps=tc_obj.get("steps", []),
                            expected_result=tc_obj.get("expected_result", ""),
                            priority=tc_obj.get("priority", "Medium"),
                        )
                        res = IntegrationTools.create_zephyr_testcase(tc)
                        executed["zephyr"].append(res)
                    except Exception as e:
                        executed["errors"].append({"error": str(e), "action": act})
                else:
                    executed["errors"].append({"unknown_action": act})
        except Exception as e:
            executed["errors"].append({"error_parsing_plan": str(e), "raw_response": resp})
        return executed

# -----------------------
# CLI Entrypoint
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Multi-Agent Test Engineering System (file ingestion + AutoGen)")
    p.add_argument("--files", nargs="+", help="List of requirement files (.txt, .docx, .pdf, .png/.jpg).", required=False)
    p.add_argument("--dir", help="Directory to scan for requirement files (optional).", required=False)
    p.add_argument("--no-integration", action="store_true", help="Do not attempt JIRA/Zephyr integration (dry run).")
    return p.parse_args()

def collect_file_list(files: Optional[List[str]], directory: Optional[str]) -> List[str]:
    collected = []
    if files:
        collected.extend(files)
    if directory:
        for entry in os.listdir(directory):
            full = os.path.join(directory, entry)
            if os.path.isfile(full) and os.path.splitext(full)[1].lower() in [".txt", ".docx", ".pdf", ".png", ".jpg", ".jpeg"]:
                collected.append(full)
    # dedupe and keep only existing
    unique = []
    for f in collected:
        if f not in unique and os.path.exists(f):
            unique.append(f)
    return unique

def main():
    args = parse_args()
    file_list = collect_file_list(args.files, args.dir)

    if not file_list:
        print("No input files provided. Use --files or --dir. Exiting.")
        sys.exit(1)

    print(f"Loading {len(file_list)} files...")
    combined_text = load_requirements_from_files(file_list)
    # simple fallback if extraction returned no useful text
    if not combined_text.strip():
        print("Warning: extracted text is empty. Check file formats or OCR installation.")
    # initialize multi-agent system
    system = TestEngineeringSystem(llm_config)

    print("\nAnalyzing requirements...")
    requirements = system.analyze_requirements(combined_text)
    print(f"Extracted {len(requirements)} requirement(s).")
    for r in requirements:
        print(json.dumps(r.to_dict(), indent=2))

    print("\nCreating test plan...")
    testplan = system.create_test_plan(requirements)
    print(json.dumps(testplan.to_dict(), indent=2))

    print("\nGenerating test cases...")
    testcases = system.generate_testcases(requirements)
    for tc in testcases:
        print(json.dumps(tc.to_dict(), indent=2))

    print("\nGenerating test data...")
    tdata = system.generate_testdata(testcases)
    for td in tdata:
        print(json.dumps(td.to_dict(), indent=2))

    if args.no_integration:
        print("\nIntegration skipped (--no-integration). Here is the plan payload.")
        payload = {
            "testplan": testplan.to_dict(),
            "testcases": [tc.to_dict() for tc in testcases],
            "testdata": [td.to_dict() for td in tdata],
        }
        print(json.dumps(payload, indent=2))
    else:
        print("\nPreparing and executing integration actions (JIRA/Zephyr) if configured...")
        executed = system.integrate_with_jira_and_zephyr(testplan, testcases, tdata)
        print("Integration results:")
        print(json.dumps(executed, indent=2))

if __name__ == "__main__":
    main()
#Quick Usage Examples

# Basic run with multiple files:

# python multiagent_test_system.py --files requirements.docx spec.pdf screenshot1.png


# Scan a directory for requirement files:

# python multiagent_test_system.py --dir ./requirements_folder


# Dry-run (no JIRA/Zephyr calls):

# python multiagent_test_system.py --files spec.pdf --no-integration