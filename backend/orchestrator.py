
"""
Agentic orchestrator: one natural-language query -> N specialised modules, run in parallel.

    [ORCH-1 PLAN]     gpt-4o + structured output decomposes the query into typed SubTasks
    [ORCH-2 EXECUTE]  each SubTask runs concurrently (asyncio.TaskGroup, 120s budget);
                      a failing module is isolated and cannot take the others down
    [ORCH-3 RESPOND]  results are shaped into a deterministic UI payload (cards, order,
                      guidance sentence) instead of a second free-form LLM call

Routing quality is measured by evals/run_evals.py --suite routing.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from modules.grad_school import GradSchoolPrep
from modules.job_search import run_job_search
from modules.resume_builder import generate_resume
from modules.skill_gap import run_skill_gap_module

# Absolute path so the orchestrator works no matter which directory it is launched from.
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
jinja_env = Environment(loader=FileSystemLoader(PROMPTS_DIR))

load_dotenv()

MAX_HISTORY_MESSAGES = 6
MAX_PROFILE_TEXT_LENGTH = 1200

MODULE_ORDER = [
    "resume_builder",
    "skill_gap",
    "interview",
    "job_search",
    "grad_school",
]

MODULE_META = {
    "resume_builder": {
        "title": "Resume Builder",
        "description": "Generate stronger resume bullets for your target role.",
        "route": None,
        "kind": "embedded",
    },
    "skill_gap": {
        "title": "Skill Gap",
        "description": "See missing skills and jump into recommended courses.",
        "route": None,
        "kind": "embedded",
    },
    "interview": {
        "title": "Interview",
        "description": "Practice role-specific interviews and get feedback.",
        "route": "/simulator",
        "kind": "redirect",
    },
    "job_search": {
        "title": "Job Search",
        "description": "Explore relevant internships and job openings.",
        "route": "/jobs",
        "kind": "redirect",
    },
    "grad_school": {
        "title": "Grad School",
        "description": "Compare programs and map your next application steps.",
        "route": "/grad",
        "kind": "redirect",
    },
}

ANIMATION_LABELS = {
    "profile": "Reading your profile",
    "planner": "Selecting the best tools",
    "resume_builder": "Generating resume bullets",
    "skill_gap": "Finding missing skills",
    "interview": "Preparing interview support",
    "job_search": "Scanning job opportunities",
    "grad_school": "Reviewing graduate school fit",
}


def truncate_text(value: Any, max_length: int = 2000) -> str:
    if value is None:
        return ""
    text = str(value)
    return text if len(text) <= max_length else text[: max_length - 1] + "…"


def compact_student_profile(profile: dict) -> dict:
    if not isinstance(profile, dict):
        return {}

    return {
        "major": profile.get("major", ""),
        "gpa": profile.get("gpa", 0.0),
        "gre": profile.get("gre", None),
        "target_job": profile.get("target_job", ""),
        "current_org": profile.get("current_org", ""),
        "current_role": profile.get("current_role", ""),
        "skills": profile.get("skills", []) or [],
        "coursework": profile.get("coursework", []) or [],
        "projects": profile.get("projects", []) or [],
        "location_preference": profile.get("location_preference", []) or [],
        "sop": truncate_text(profile.get("sop", ""), MAX_PROFILE_TEXT_LENGTH),
        "resumeText": truncate_text(profile.get("resumeText", ""), MAX_PROFILE_TEXT_LENGTH),
    }


def compact_history(history: list) -> list:
    if not isinstance(history, list):
        return []

    trimmed = history[-MAX_HISTORY_MESSAGES:]
    return [
        {
            "role": m.get("role", "human") if isinstance(m, dict) else "human",
            "message": truncate_text(m.get("message", "") if isinstance(m, dict) else str(m), 1200),
        }
        for m in trimmed
    ]


class ResumeBuilderInput(BaseModel):
    target_role: str = ""
    skills: List[str] = Field(default_factory=list)
    coursework: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    major: str = ""


class JobSearchInput(BaseModel):
    skills: List[str] = Field(default_factory=list)
    major: str = ""
    coursework: List[str] = Field(default_factory=list)
    location_preference: List[str] = Field(default_factory=list)
    target_role: str = ""


class GradSchoolInput(BaseModel):
    major: str = ""
    gpa: float = 0.0
    gre: int = 0
    coursework: List[str] = Field(default_factory=list)


class SkillGapInput(BaseModel):
    target_role: str = ""
    skills: List[str] = Field(default_factory=list)


class EmptyInput(BaseModel):
    pass


# [ORCH-1 PLAN] The planner's output contract. `with_structured_output(Decomposition)`
# turns this Pydantic schema into a JSON schema / function-calling spec, so the LLM can
# only answer with known modules and typed inputs; no regex parsing of free text.
class SubTask(BaseModel):
    module: str = Field(description="The target module name")
    inputs: Union[
        ResumeBuilderInput,
        JobSearchInput,
        GradSchoolInput,
        SkillGapInput,
        EmptyInput,
    ] = Field(description="Structured input specified to each module", discriminator=None)


class Decomposition(BaseModel):
    sub_tasks: List[SubTask]


async def call_module(module_name: str, input_data: Any):
    if module_name == "resume_builder":
        enriched_profile = {
            "skills": getattr(input_data, "skills", []) or [],
            "coursework": getattr(input_data, "coursework", []) or [],
            "projects": getattr(input_data, "projects", []) or [],
            "major": getattr(input_data, "major", "") or "",
            "target_role": getattr(input_data, "target_role", "") or "",
            "resumeText": getattr(input_data, "resumeText", "") or "",
            "current_role": getattr(input_data, "current_role", "") or "",
            "current_org": getattr(input_data, "current_org", "") or "",
        }
        loop = asyncio.get_running_loop()
        target_role = getattr(input_data, "target_role", "") or ""
        normalized_targets = [item.strip() for item in str(target_role).split(",") if item.strip()]

        return await loop.run_in_executor(
            None,
            generate_resume,
            enriched_profile,
            normalized_targets if normalized_targets else None,
        )

    if module_name == "job_search":
        student_profile = {
            "skills": getattr(input_data, "skills", []) or [],
            "major": getattr(input_data, "major", "") or "",
            "coursework": getattr(input_data, "coursework", []) or [],
            "location_preference": getattr(input_data, "location_preference", []) or [],
            "target_role": getattr(input_data, "target_role", "") or "",
        }
        return await run_job_search(student_profile)

    if module_name == "grad_school":
        grad_profile = {
            "major": getattr(input_data, "major", "") or "",
            "gpa": getattr(input_data, "gpa", 0.0) or 0.0,
            "gre": getattr(input_data, "gre", 0) or 0,
            "coursework": getattr(input_data, "coursework", []) or [],
        }
        module = GradSchoolPrep()
        return await module.run(grad_profile)

    if module_name == "skill_gap":
        enriched_profile = {
            "skills": getattr(input_data, "skills", []) or [],
        }
        return await run_skill_gap_module(getattr(input_data, "target_role", ""), enriched_profile)

    if module_name == "interview":
        return {
            "module": "interview",
            "message": "Open the interview tool to practice role-specific questions and get feedback.",
        }

    return {"error": f"Unknown module: {module_name}"}


async def safe_execute_module(module_name: str, input_data: Any) -> Dict[str, Any]:
    try:
        result = await call_module(module_name, input_data)
        return {"module": module_name, "result": result, "error": None}
    except Exception as exc:
        return {"module": module_name, "result": {"error": str(exc)}, "error": str(exc)}


def build_animation_steps(triggered_modules: List[str]) -> List[dict]:
    steps = [
        {"id": "profile", "label": ANIMATION_LABELS["profile"]},
        {"id": "planner", "label": ANIMATION_LABELS["planner"]},
    ]
    seen = {"profile", "planner"}

    for module_name in triggered_modules:
        if module_name in ANIMATION_LABELS and module_name not in seen:
            steps.append({"id": module_name, "label": ANIMATION_LABELS[module_name]})
            seen.add(module_name)

    return steps


def build_guidance_sentence(triggered_modules: List[str], results: Dict[str, Any]) -> str:
    if "resume_builder" in triggered_modules and "skill_gap" in triggered_modules:
        return "I refreshed your resume bullets and found the main skills to build next—review the highlighted cards below, then open another tool when you want to keep going."
    if "resume_builder" in triggered_modules:
        return "Your updated resume bullets are ready below—review them, copy what you need, and open another tool if you want help with your next step."
    if "skill_gap" in triggered_modules:
        return "I found the biggest skills to strengthen next—review the highlighted card below and use the course buttons to start closing the gap."
    if "interview" in triggered_modules:
        return "Your next best step is interview practice—open the highlighted tool below to start a guided mock interview."
    if "job_search" in triggered_modules:
        return "Your next best step is exploring roles—open the highlighted job search tool below to browse matching opportunities."
    if "grad_school" in triggered_modules:
        return "Your next best step is planning for graduate programs—open the highlighted tool below to continue."
    if results:
        return "I found the most relevant tools for your request—start with the highlighted cards below."
    return "Pick any tool below to keep building toward your next career step."


def pair_skill_gap_items(skill_gap_result: Dict[str, Any]) -> List[dict]:
    gaps = skill_gap_result.get("gaps") or []
    recommendations = skill_gap_result.get("recommendations") or []
    paired_items: List[dict] = []

    for index, gap in enumerate(gaps):
        course = recommendations[index] if index < len(recommendations) else None
        paired_items.append(
            {
                "skill": gap.get("skill", ""),
                "gap_score": gap.get("gap_score", 1),
                "description": gap.get("description", ""),
                "course_label": (course or {}).get("course", ""),
                "course_provider": (course or {}).get("platform", ""),
                "course_url": (course or {}).get("url", ""),
            }
        )

    return paired_items


def build_ui_cards(triggered_modules: List[str], results: Dict[str, Any]) -> Dict[str, dict]:
    cards: Dict[str, dict] = {}

    for module_name in MODULE_ORDER:
        meta = MODULE_META[module_name]
        is_active = module_name in triggered_modules
        card = {
            "module": module_name,
            "title": meta["title"],
            "description": meta["description"],
            "route": meta["route"],
            "kind": meta["kind"],
            "active": is_active,
        }

        if module_name == "resume_builder":
            bullets = results.get("resume_builder") if isinstance(results.get("resume_builder"), list) else []
            cleaned_bullets = [str(b).strip() for b in bullets if str(b).strip()]
            card.update(
                {
                    "bullets": cleaned_bullets,
                    "copy_text": "\n".join(cleaned_bullets),
                    "has_content": bool(cleaned_bullets),
                }
            )
        elif module_name == "skill_gap":
            skill_gap_result = results.get("skill_gap") if isinstance(results.get("skill_gap"), dict) else {}
            items = pair_skill_gap_items(skill_gap_result)
            card.update(
                {
                    "target_occupation": skill_gap_result.get("target_occupation", ""),
                    "items": items,
                    "has_content": bool(items),
                }
            )
        else:
            card.update({"has_content": is_active})

        cards[module_name] = card

    return cards


def build_tool_order(triggered_modules: List[str]) -> List[str]:
    active_first = [module for module in triggered_modules if module in MODULE_ORDER]
    remaining = [module for module in MODULE_ORDER if module not in active_first]
    return active_first + remaining


def build_ui_payload(query: str, triggered_modules: List[str], results: Dict[str, Any]) -> dict:
    cards = build_ui_cards(triggered_modules, results)
    tool_order = build_tool_order(triggered_modules)

    return {
        "hero_title": "What do you want to work on today?",
        "hero_placeholder": "Ask Elevatr to strengthen your resume, find skill gaps, or point you to the right tool…",
        "guidance_sentence": build_guidance_sentence(triggered_modules, results),
        "tool_order": tool_order,
        "animation_steps": build_animation_steps(triggered_modules),
        "cards": {module_name: cards[module_name] for module_name in tool_order},
        "latest_query": query,
    }


async def decompose(query: str, student_profile: dict, history: Optional[list] = None) -> Decomposition:
    """[ORCH-1 PLAN] Ask the LLM which modules to call and with what inputs."""
    history = compact_history(history or [])
    # Context budgeting: only the last MAX_HISTORY_MESSAGES turns and truncated free-text
    # fields are sent, which bounds prompt size (cost + latency) regardless of user data.
    compacted_profile = compact_student_profile(student_profile)

    decomp_template = jinja_env.get_template("decomp_prompt.j2")
    decomp_prompt = decomp_template.render()

    model = init_chat_model(model="gpt-4o")
    conversation = [SystemMessage(decomp_prompt)]
    conversation.append(SystemMessage(f"Student Profile: {compacted_profile}"))

    for message in history:
        if message.get("role") == "human":
            conversation.append(HumanMessage(message.get("message", "")))
        else:
            conversation.append(AIMessage(message.get("message", "")))

    conversation.append(HumanMessage(f"Query: {truncate_text(query, 1200)}"))

    structured_model = model.with_structured_output(Decomposition)
    # The LangChain call is blocking; run it in a thread so the event loop keeps serving requests.
    return await asyncio.to_thread(structured_model.invoke, conversation)


async def run(user_id: str, query: str, student_profile: dict, history: Optional[list] = None):
    response = await decompose(query, student_profile, history)
    # De-duplicate while preserving the planner's order (it is also the UI card order).
    triggered_modules = list(dict.fromkeys(task.module for task in response.sub_tasks))

    # [ORCH-2 EXECUTE] Fan out. safe_execute_module never raises, so one module's
    # exception becomes an error field in its own result instead of cancelling the group.
    execution_list = []
    try:
        async with asyncio.timeout(120.0):
            async with asyncio.TaskGroup() as tg:
                for task in response.sub_tasks:
                    execution_task = tg.create_task(safe_execute_module(task.module, task.inputs))
                    execution_list.append(execution_task)
    except TimeoutError as exc:
        raise Exception("Module execution timed out.") from exc

    results: Dict[str, Any] = {}
    for task in execution_list:
        payload = task.result()
        results[payload["module"]] = payload["result"]

    # [ORCH-3 RESPOND] Deterministic shaping: no extra LLM call, so no extra latency or hallucination.
    ui = build_ui_payload(query, triggered_modules, results)
    guidance_sentence = ui["guidance_sentence"]

    return {
        "query": query,
        "modules_triggered": triggered_modules,
        "results": results,
        "answer": guidance_sentence,
        "guidance_sentence": guidance_sentence,
        "ui": ui,
    }


if __name__ == "__main__":
    async def _demo():
        output = await run(
            user_id="test",
            query="Help me find skill gaps and generate resume bullets for software engineering",
            student_profile={"skills": ["Python"], "major": "CS", "gpa": 3.5},
            history=[],
        )
        print(json.dumps(output, indent=2))

    asyncio.run(_demo())
