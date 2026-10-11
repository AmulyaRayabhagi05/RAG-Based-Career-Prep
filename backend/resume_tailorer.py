#!/usr/bin/env python3
"""
resume_tailorer.py
──────────────────
Rewrites resume bullets using the WHO Method (What / How / Outcome) and
generates a one-page PDF via xhtml2pdf + Jinja2.

Usage:
    python resume_tailorer.py

Dependencies:
    pip install openai jinja2 xhtml2pdf python-dotenv
"""

import json
import os
import re
import sys

from llm import get_openai_client, model_name
from jinja2 import Environment, FileSystemLoader
from xhtml2pdf import pisa
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI client ──────────────────────────────────────────────────────────────
client = get_openai_client()

BULLET_SYSTEM_PROMPT = """You are a professional resume writer. Your job is to improve resume bullets
using the WHO Method while staying 100% truthful to what the candidate actually did.

WHO Method:
  WHAT     – Strong, specific action verb.
  HOW      – The actual tool, method, or technology they used (from the original bullet).
  OUTCOME  – A quantified result (%, time, $, scale) — estimate realistically if unknown.

CRITICAL RULES:
- Do NOT change the field, role, or responsibilities described in the original bullet.
  If they worked in IT compliance → keep it IT compliance.
  If they built a web app → keep it a web app. Never convert it to something else.
- You MAY naturally emphasise aspects of their real work that overlap with the job
  description themes (e.g. data integrity, automation, security awareness) — only if
  those aspects are already implied by the original bullet.
- Never fabricate tools, responsibilities, or domain experience not in the original.
- Output ONLY the rewritten bullet. No preamble, no quotes, no explanation.
- Every bullet must contain at least one number or measurable metric.
  Use a realistic estimate if no exact figure is known (e.g. "~30%", "50+ records").
- One sentence, under 30 words where possible.
- Do not start two bullets for the same entry with the same verb.
"""


def call_openai(prompt: str, system: str = BULLET_SYSTEM_PROMPT) -> str | None:
    """Call GPT-4o and return the stripped text response."""
    try:
        response = client.chat.completions.create(
            model=model_name("large"),
            max_tokens=300,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠  OpenAI API error: {e}")
        return None


# ── Keyword extraction ─────────────────────────────────────────────────────────

def extract_keywords_from_jd(job_description: str) -> list[str]:
    """Ask GPT to pull the 8 most important themes/skills from the JD."""
    prompt = f"""Read this job description and return a JSON array of the 8 most important
technical skills, tools, or domain themes mentioned.
Return ONLY valid JSON — an array of strings, no preamble or markdown fences.

Job description:
{job_description[:1500]}"""

    result = call_openai(
        prompt,
        system="You extract structured data from job descriptions. Return only valid JSON arrays."
    )

    if not result:
        return []

    clean = re.sub(r"```(?:json)?|```", "", result).strip()
    try:
        kws = json.loads(clean)
        if isinstance(kws, list):
            return [str(k) for k in kws]
    except json.JSONDecodeError:
        pass
    return []


# ── Bullet rewriting ───────────────────────────────────────────────────────────

def rewrite_bullet(
    original: str,
    job_description: str,
    keywords: list[str],
    item_context: str = "",
) -> str:
    kw_str = ", ".join(keywords[:6]) if keywords else "general professional skills"

    prompt = f"""Rewrite this resume bullet using the WHO Method.

IMPORTANT: Keep exactly the same role/field/responsibilities as the original.
Only emphasise overlapping aspects — do not invent new experience.

Original bullet:
  "{original}"

Role/project context (for tone only, not to override the original):
  {item_context}

Job description themes to lean toward (only where already present in the original):
  {kw_str}

Rewritten bullet (WHO Method, truthful, quantified, one sentence):"""

    result = call_openai(prompt)

    if result and not re.search(r"\d", result):
        result = result.rstrip(".") + ", improving efficiency by ~20%."

    return result or original


def rewrite_bullets(items: list[dict], job_description: str, keywords: list[str]) -> None:
    for item in items:
        context = item.get("title") or item.get("name") or ""
        if item.get("company"):
            context += f" at {item['company']}"

        rewritten = []
        for bullet in item.get("bullet_points", []):
            new = rewrite_bullet(bullet, job_description, keywords, context)
            print(f"    ✓ {new[:90]}{'…' if len(new) > 90 else ''}")
            rewritten.append(new)
        item["bullet_points"] = rewritten


# ── Field-agnostic skill categorisation ───────────────────────────────────────

def categorise_skills(skills: list[str]) -> dict[str, list[str]]:
    # Checked in order — Cloud/DevOps first to prevent Azure/Docker/K8s
    # from falling into Programming
    RULES: list[tuple[str, set[str]]] = [
        ("Cloud & DevOps", {
            "aws", "azure", "gcp", "google cloud", "heroku",
            "kubernetes", "docker", "terraform", "ansible",
            "jenkins", "github actions", "ci/cd", "cloudformation",
            "lambda", "vercel", "netlify", "openshift", "helm",
            "s3", "ec2", "rds", "elastic beanstalk",
        }),
        ("Data & Analytics", {
            "tableau", "power bi", "powerbi", "excel", "jupyter",
            "matplotlib", "seaborn", "spark", "hadoop", "kafka",
            "databricks", "looker", "sas", "spss", "stata",
            "snowflake", "dbt", "pandas", "numpy", "scipy",
        }),
        ("Frameworks & Libraries", {
            "react", "angular", "vue", "node", "django", "flask",
            "spring", "express", "fastapi", "next", "nuxt", "rails",
            "laravel", "tensorflow", "pytorch", "keras",
            "scikit", "sklearn", "opencv",
        }),
        ("Tools & Platforms", {
            "git", "github", "gitlab", "jira", "confluence", "postman",
            "vscode", "intellij", "eclipse", "xcode", "figma", "sketch",
            "mongodb", "postgresql", "mysql", "sqlite", "redis",
            "elasticsearch", "graphql", "rest", "soap",
            "linux", "unix", "windows server",
            "tia portal", "siemens", "profinet", "labview",
            "solidworks", "autocad", "ansys", "simulink",
            "wireshark", "burp suite", "metasploit", "nmap",
            "splunk", "siem",
        }),
        ("Programming", {
            "python", "java", "javascript", "typescript",
            "c++", "c#", "golang", "rust", "ruby",
            "php", "swift", "kotlin", "scala", "r", "matlab",
            "bash", "shell", "sql", "html", "css",
            "assembly", "verilog", "vhdl", "ladder logic",
            "structured text", "plc",
        }),
    ]

    buckets: dict[str, list[str]] = {name: [] for name, _ in RULES}
    buckets["Other"] = []

    for skill in skills:
        lower = " " + skill.lower() + " "
        placed = False
        for cat_name, keywords in RULES:
            if any(kw in lower for kw in keywords):
                buckets[cat_name].append(skill)
                placed = True
                break
        if not placed:
            buckets["Other"].append(skill)

    return {k: v for k, v in buckets.items() if v}


# ── PDF generation ─────────────────────────────────────────────────────────────

def generate_pdf(
    resume_data: dict,
    template_file: str = "resume_template.html",
    output_file: str = "tailored_resume.pdf",
) -> None:
    template_dir  = os.path.dirname(os.path.abspath(template_file)) or "."
    template_name = os.path.basename(template_file)

    env      = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)
    html_out = template.render(**resume_data)

    with open(output_file, "wb") as f:
        result = pisa.CreatePDF(html_out, dest=f)

    if result.err:
        print(f"  ⚠  PDF errors (code {result.err}).")
    else:
        print(f"  ✨ PDF saved → {output_file}")


# ── Optional summary ───────────────────────────────────────────────────────────

def generate_summary_if_empty(resume_data: dict, job_description: str) -> None:
    """Leave summary blank (section hidden) unless GENERATE_SUMMARY=true in .env."""
    if os.getenv("GENERATE_SUMMARY", "false").lower() != "true":
        resume_data.setdefault("summary", "")
        return

    if resume_data.get("summary", "").strip():
        return

    print("  📝 Generating optional summary …")
    edu        = resume_data.get("education", [])
    degree_str = f"{edu[0]['degree']} in {edu[0]['major']}" if edu else "a relevant degree"
    exp        = resume_data.get("experience", [])
    role_str   = exp[0]["title"] if exp else "their field"

    prompt = f"""Write a 2-sentence professional resume summary for {resume_data.get('name', 'the candidate')},
who holds {degree_str} and works as {role_str}.
Tailor toward this role without misrepresenting their background: {job_description[:400]}
Output ONLY the summary text, no labels."""

    resume_data["summary"] = call_openai(
        prompt, system="You write concise, honest, impactful resume summaries."
    ) or ""


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        with open("resume_data.json") as f:
            resume_data: dict = json.load(f)
        with open("job_description.txt") as f:
            job_description: str = f.read()
    except FileNotFoundError as e:
        print(f"❌  Missing file: {e.filename}")
        sys.exit(1)

    print("🔍  Extracting keywords from job description …")
    keywords = extract_keywords_from_jd(job_description)
    print(f"    → {', '.join(keywords) or '(none extracted)'}")

    print("\n⚡  Rewriting experience bullets …")
    rewrite_bullets(resume_data.get("experience", []), job_description, keywords)

    print("\n🛠   Rewriting project bullets …")
    rewrite_bullets(resume_data.get("projects", []), job_description, keywords)

    generate_summary_if_empty(resume_data, job_description)

    flat_skills = resume_data.get("skills", [])
    resume_data["categorized_skills"] = categorise_skills(flat_skills)

    with open("tailored_resume_data.json", "w") as f:
        json.dump(resume_data, f, indent=2)
    print("\n✅  Saved → tailored_resume_data.json")

    print("📄  Rendering PDF …")
    generate_pdf(resume_data, template_file="resume_template.html")


if __name__ == "__main__":
    main()