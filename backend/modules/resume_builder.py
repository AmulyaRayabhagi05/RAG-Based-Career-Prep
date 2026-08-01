import json
import os
import sys
import random
import re

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from rag.retriever import retrieve_for_resume
except ImportError:
    from ..rag.retriever import retrieve_for_resume

VERB_MAP = {
    "python": ["Developed", "Automated", "Built", "Implemented"],
    "java": ["Engineered", "Designed", "Built", "Developed"],
    "sql": ["Queried", "Optimized", "Designed", "Analyzed"],
    "machine learning": ["Trained", "Developed", "Implemented", "Evaluated"],
    "ml": ["Trained", "Built", "Implemented", "Deployed"],
    "data": ["Analyzed", "Processed", "Visualized", "Built"],
    "database": ["Designed", "Optimized", "Maintained", "Queried"],
    "react": ["Built", "Developed", "Designed", "Implemented"],
    "javascript": ["Developed", "Built", "Implemented", "Created"],
    "cloud": ["Deployed", "Managed", "Architected", "Configured"],
    "aws": ["Deployed", "Configured", "Managed", "Architected"],
    "api": ["Designed", "Built", "Integrated", "Developed"],
    "algorithm": ["Designed", "Optimized", "Implemented", "Analyzed"],
    "default": ["Developed", "Implemented", "Built", "Designed", "Created"]
}

TEMPLATES = [
    "{verb} {skill}-based solution for {task}, directly applicable to {title} roles.",
    "{verb} end-to-end pipeline using {skill} to {task}, improving workflow efficiency.",
    "{verb} {title}-focused project leveraging {skill} to {task}.",
    "{verb} and optimized {skill} workflows to {task}, strengthening {title} competencies.",
    "{verb} scalable system using {skill} to support {task} in a {title} context."
]

TASK_PHRASES = [
    "analyze large datasets",
    "automate workflows",
    "build scalable systems",
    "develop APIs",
    "optimize performance",
    "solve complex problems"
]


def normalize_skill(s: str) -> str:
    s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'[^\w\s]', '', s)
    return s.strip().lower()


def get_verb(skill: str) -> str:
    for key, verbs in VERB_MAP.items():
        if key in skill.lower():
            return random.choice(verbs)
    return random.choice(VERB_MAP["default"])


def extract_task(text: str) -> str:
    for line in text.split("\n"):
        if line.lower().startswith("tasks:"):
            return line.split(":", 1)[-1].strip()[:80]
    return random.choice(TASK_PHRASES)


def extract_requirements(job_chunks):
    return [
        {"title": chunk.get("title", ""), "text": chunk.get("text", "")}
        for chunk in job_chunks
    ]


def map_student_to_jobs(student, job_data):
    keywords = set(
        [normalize_skill(s) for s in student.get("skills", [])] +
        [normalize_skill(c) for c in student.get("coursework", [])] +
        [normalize_skill(p) for p in student.get("projects", [])] +
        [normalize_skill(student.get("major", ""))]
    )

    matched = []
    for job in job_data:
        text = job["text"].lower()
        overlap = [k for k in keywords if k and k in text]
        if overlap:
            job["matched_keywords"] = overlap
            matched.append(job)

    return matched


def generate_bullets(student, matched_jobs):
    bullets = []
    seen = set()
    skills = student.get("skills", []) or student.get("coursework", []) or student.get("projects", []) or ["software development"]
    target_focus = student.get("target_role") or student.get("major") or "target role"

    for job in matched_jobs[:5]:
        title = job["title"]
        skills_to_use = job.get("matched_keywords", [])[:2] or [normalize_skill(s) for s in skills[:2]]

        for skill in skills_to_use:
            verb = get_verb(skill)
            task = extract_task(job["text"])
            template = random.choice(TEMPLATES)

            bullet = template.format(
                verb=verb,
                skill=skill.title(),
                title=title,
                task=task
            )

            if bullet not in seen:
                seen.add(bullet)
                bullets.append(bullet)

    if not bullets:
        fallback_terms = [normalize_skill(s) for s in skills[:4] if normalize_skill(s)]
        if not fallback_terms:
            fallback_terms = ["software development"]

        for term in fallback_terms:
            verb = get_verb(term)
            task = random.choice(TASK_PHRASES)
            bullet = f"{verb} {term.title()} initiatives to {task}, aligning with {target_focus} goals."
            if bullet not in seen:
                seen.add(bullet)
                bullets.append(bullet)

    return bullets


def generate_resume(student, target_titles=None):
    if isinstance(target_titles, list):
        all_chunks = []
        for title in target_titles:
            retrieved = retrieve_for_resume(student, target_title=title)
            all_chunks.extend(retrieved["job_chunks"])

        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk["text"] not in seen:
                seen.add(chunk["text"])
                unique_chunks.append(chunk)

        job_chunks = unique_chunks
    else:
        job_chunks = retrieve_for_resume(student, target_title=None)["job_chunks"]

    job_data = extract_requirements(job_chunks)
    matched = map_student_to_jobs(student, job_data)

    if not matched:
        matched = job_data[:5]

    return generate_bullets(student, matched)


if __name__ == "__main__":
    from get_student_profile import get_student_profile
    student, targets = get_student_profile()
    bullets = generate_resume(student, targets)

    print("\n=== YOUR BULLETS ===")
    for i, b in enumerate(bullets, 1):
        print(f"{i}. {b}")

    with open("resume_output.json", "w") as f:
        json.dump({"bullets": bullets}, f, indent=2)

    print("Saved to resume_output.json")