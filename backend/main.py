import sys
import os
import io
import tempfile
sys.path.insert(0, os.path.dirname(__file__))
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, Annotated
import asyncio
import jwt
import re
import json
import copy
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

from database import mongo, memory
from orchestrator import run as run_orchestrator
from modules.resume_parser import get_pdf_text, parse_resume
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from auth import get_current_user
from modules.grad_router import router as grad_router

# Resume tailoring imports
from openai import AsyncOpenAI
from llm import get_async_openai_client, model_name
from jinja2 import Environment, FileSystemLoader
from xhtml2pdf import pisa

load_dotenv()

# ---- CONFIG ----
SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(title="Elevatr API")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from modules.interview_router import router as interview_router
from modules.job_searchv2 import run_job_search
from modules.grad_router import router
app.include_router(interview_router)
app.include_router(router)


@app.get("/speech-token")
async def get_speech_token():
    return {
        "token": os.getenv("VITE_AZURE_SPEECH_KEY"),
        "region": os.getenv("VITE_AZURE_SPEECH_REGION")
    }

# ---- PASSWORD & JWT HELPERS ----

password_hash = PasswordHash.recommended()

def verify_password(plain_password, hashed_password):
    return password_hash.verify(plain_password, hashed_password)

def get_password_hash(password):
    return password_hash.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def authenticate_user(email:str, password:str):
    user = await mongo.get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return user


# ---- SCHEMAS ----

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class HistoryMessage(BaseModel):
    role: str
    message: str

class QueryRequest(BaseModel):
    query: str
    history: list[HistoryMessage] | None = None

class JobSearchRequest(BaseModel):
    skills: list[str] = []
    major: str = ""
    coursework: list[str] = []
    location_preference: list[str] = []
    target_role: str = ""
    query: str = ""
    sources: list[str] = []
    company: str = ""
    location: str = ""
    remote_only: bool = False
    posted_within_days: int = 0

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: str | None = None


# ---- STARTUP / SHUTDOWN ----

@app.on_event("startup")
async def startup():
    await mongo.connect()

@app.on_event("shutdown")
async def shutdown():
    mongo.client.close()


# ---- TOKEN ROUTE FOR TESTING ----
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]) -> Token:
    user = await authenticate_user(form_data.username, form_data.password)
    access_token = create_access_token(
        data={"sub": str(user["_id"])},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return Token(access_token=access_token, token_type="bearer")

# ---- AUTH ROUTES ----
@app.post("/auth/register", response_model=Token)
async def register(body: RegisterRequest):
    existing = await mongo.get_user_by_email(body.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed = get_password_hash(body.password)
    result = await mongo.register(body.name, body.email, hashed)

    user_id = str(result.inserted_id)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_id}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


@app.post("/auth/login", response_model=Token)
async def login(body: LoginRequest):
    user = await authenticate_user(body.email, body.password)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user["_id"])}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


# ---- PROFILE ROUTES ----

@app.get("/profile")
async def get_profile(user_id: str = Depends(get_current_user)):
    profile = await mongo.get_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@app.put("/profile")
async def update_profile(profile: dict, user_id: str = Depends(get_current_user)):
    await mongo.update_profile(user_id, profile)
    return {"message": "Profile updated"}

@app.post("/job-search")
async def job_search(body: JobSearchRequest):
    return run_job_search(body.dict())

# ---- RESUME PARSER HELPERS ----

def _normalize_gpa(value: str | float | None) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if "/" in text:
            text = text.split("/")[0].strip()
        return float(text)
    except ValueError:
        return None


def _flatten_skills(skills: dict | list | None) -> list[str]:
    if not skills:
        return []
    if isinstance(skills, list):
        return [s for s in skills if isinstance(s, str)]
    result: list[str] = []
    if isinstance(skills, dict):
        for section in skills.values():
            if isinstance(section, list):
                result.extend([str(item).strip() for item in section if item])
            elif isinstance(section, str):
                result.extend([s.strip() for s in re.split(r"[,;|]", section) if s.strip()])
    return list(dict.fromkeys(result))


def _serialize_experience(experience: list[dict]) -> str:
    if not experience:
        return ""
    lines: list[str] = []
    for item in experience:
        header = " ".join(str(item.get(key, "")).strip() for key in ["title", "company"] if item.get(key))
        if header:
            lines.append(header)
        for bullet in item.get("bullet_points", []):
            if bullet:
                lines.append(f"- {bullet}")
        if item.get("start_date") or item.get("end_date"):
            dates = " - ".join(str(item.get(key, "")).strip() for key in ["start_date", "end_date"] if item.get(key))
            lines.append(dates)
        lines.append("")
    return "\n".join(line for line in lines if line.strip())


def _serialize_projects(projects: list[dict]) -> str:
    if not projects:
        return ""
    lines: list[str] = []
    for item in projects:
        if item.get("name"):
            lines.append(str(item.get("name")).strip())
        for bullet in item.get("bullet_points", []):
            if bullet:
                lines.append(f"- {bullet}")
        if item.get("technologies"):
            techs = item.get("technologies")
            if isinstance(techs, list):
                lines.append("Technologies: " + ", ".join([str(t).strip() for t in techs if t]))
        if item.get("url"):
            lines.append(f"URL: {item.get('url')}")
        lines.append("")
    return "\n".join(line for line in lines if line.strip())


@app.post("/profile/parse_resume")
async def parse_resume_endpoint(file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    content = await file.read()
    text = ""
    filename = file.filename or "resume"
    lower = filename.lower()

    if lower.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp.flush()
            text = get_pdf_text(tmp.name)
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    else:
        try:
            text = content.decode("utf-8")
        except Exception:
            text = content.decode("latin-1", errors="ignore")

    parsed = parse_resume(text)

    profile = await mongo.get_profile(user_id) or {}

    education = parsed.get("education") or []
    parsed_major = ""
    parsed_gpa = None
    parsed_coursework: list[str] = []
    for edu in education:
        if not parsed_major:
            parsed_major = edu.get("major") or ""
        if parsed_gpa is None:
            parsed_gpa = _normalize_gpa(edu.get("gpa"))
        for course in (edu.get("relevant_coursework") or []):
            if course and course not in parsed_coursework:
                parsed_coursework.append(course)

    # experience_raw and projects_raw come from parse_resume directly
    experience_raw = parsed.get("experience_raw") or parsed.get("experience") or []
    projects_raw   = parsed.get("projects_raw")   or parsed.get("projects")   or []

    # Grab current role and org straight from the first experience entry — no derivation logic
    first_exp    = experience_raw[0] if experience_raw else {}
    current_role = first_exp.get("title", "")   or profile.get("current_role", "")
    current_org  = first_exp.get("company", "") or profile.get("current_org", "")

    updated = {
        "resumeText":     text,
        "resumeFileName": filename,
        "name":           parsed.get("name") or profile.get("name", ""),
        "major":          parsed_major or profile.get("major", ""),
        "gpa":            parsed_gpa or profile.get("gpa", 0.0),
        "skills":         _flatten_skills(parsed.get("skills")) or profile.get("skills", []),
        "coursework":     parsed_coursework or profile.get("coursework", []),
        "experience":     _serialize_experience(experience_raw) or profile.get("experience", ""),
        "projects":       _serialize_projects(projects_raw) or profile.get("projects", ""),
        "experience_raw": experience_raw,
        "projects_raw":   projects_raw,
        "current_role":   current_role,
        "current_org":    current_org,
        "phone":          parsed.get("phone")    or profile.get("phone"),
        "linkedin":       parsed.get("linkedin") or profile.get("linkedin"),
        "github":         parsed.get("github")   or profile.get("github"),
        "location":       parsed.get("location") or profile.get("location"),
        "education":      education or profile.get("education", []),
        "certifications": parsed.get("certifications") or profile.get("certifications", []),
        "achievements":   parsed.get("achievements")   or profile.get("achievements", []),
        "leadership":     parsed.get("leadership")     or profile.get("leadership", []),
    }

    if profile.get("email"):
        updated["email"] = profile["email"]

    await mongo.update_profile(user_id, updated)
    return await mongo.get_profile(user_id)


# ════════════════════════════════════════════════════════════════════════════
# RESUME TAILORING — helpers (async, no OpenAI client at module level)
# ════════════════════════════════════════════════════════════════════════════

_BULLET_SYSTEM = """You are a professional resume writer. Rewrite resume bullets using the WHO Method:
  WHAT   – Strong, specific action verb.
  HOW    – The actual tool, method, or technology from the original bullet.
  OUTCOME – A quantified result (%, time, $, scale). Estimate realistically if unknown.

RULES:
- Keep exactly the same role/field/responsibilities. Never fabricate new experience.
- Output ONLY the rewritten bullet. No preamble, no quotes.
- Must contain at least one number or measurable metric.
- One sentence, under 30 words where possible.
- Do not start two bullets for the same entry with the same verb."""

_SKILL_CATEGORIES: list[tuple[str, set[str]]] = [
    ("Cloud & DevOps", {
        "aws","azure","gcp","google cloud","heroku","kubernetes","docker","terraform",
        "ansible","jenkins","github actions","ci/cd","cloudformation","lambda","vercel",
        "netlify","openshift","helm","s3","ec2","rds","elastic beanstalk",
    }),
    ("Data & Analytics", {
        "tableau","power bi","powerbi","excel","jupyter","matplotlib","seaborn","spark",
        "hadoop","kafka","databricks","looker","sas","spss","stata","snowflake","dbt",
        "pandas","numpy","scipy",
    }),
    ("Frameworks & Libraries", {
        "react","angular","vue","node","django","flask","spring","express","fastapi",
        "next","nuxt","rails","laravel","tensorflow","pytorch","keras","scikit","sklearn","opencv",
    }),
    ("Tools & Platforms", {
        "git","github","gitlab","jira","confluence","postman","vscode","intellij","eclipse",
        "xcode","figma","sketch","mongodb","postgresql","mysql","sqlite","redis",
        "elasticsearch","graphql","rest","soap","linux","unix","windows server",
        "tia portal","siemens","profinet","labview","solidworks","autocad","ansys",
        "simulink","wireshark","burp suite","metasploit","nmap","splunk","siem",
    }),
    ("Programming", {
        "python","java","javascript","typescript","c++","c#","golang","rust","ruby",
        "php","swift","kotlin","scala","r","matlab","bash","shell","sql","html","css",
        "assembly","verilog","vhdl","ladder logic","structured text","plc",
    }),
]

def _categorise_skills(skills: list[str]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {name: [] for name, _ in _SKILL_CATEGORIES}
    buckets["Other"] = []
    for skill in skills:
        lower = " " + skill.lower() + " "
        placed = False
        for cat_name, keywords in _SKILL_CATEGORIES:
            if any(kw in lower for kw in keywords):
                buckets[cat_name].append(skill)
                placed = True
                break
        if not placed:
            buckets["Other"].append(skill)
    return {k: v for k, v in buckets.items() if v}


async def _call_openai_async(client: AsyncOpenAI, prompt: str, system: str = _BULLET_SYSTEM) -> str | None:
    try:
        resp = await client.chat.completions.create(
            model=model_name("large"),
            max_tokens=300,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI error: {e}")
        return None


async def _extract_keywords(client: AsyncOpenAI, job_description: str) -> list[str]:
    prompt = (
        "Read this job description and return a JSON array of the 8 most important "
        "technical skills, tools, or domain themes. Return ONLY valid JSON — "
        "an array of strings, no preamble or markdown fences.\n\n"
        f"Job description:\n{job_description[:1500]}"
    )
    result = await _call_openai_async(
        client, prompt,
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


async def _rewrite_bullet(
    client: AsyncOpenAI,
    original: str,
    keywords: list[str],
    item_context: str = "",
) -> str:
    kw_str = ", ".join(keywords[:6]) if keywords else "general professional skills"
    prompt = (
        f'Rewrite this resume bullet using the WHO Method.\n\n'
        f'Keep exactly the same role/field/responsibilities. Only emphasise '
        f'overlapping aspects — do not invent new experience.\n\n'
        f'Original bullet:\n  "{original}"\n\n'
        f'Role/project context (for tone only):\n  {item_context}\n\n'
        f'Job description themes (only where already present in the original):\n  {kw_str}\n\n'
        f'Rewritten bullet (WHO Method, truthful, quantified, one sentence):'
    )
    result = await _call_openai_async(client, prompt)
    if result and not re.search(r"\d", result):
        result = result.rstrip(".") + ", improving efficiency by ~20%."
    return result or original


async def _rewrite_all_bullets(
    client: AsyncOpenAI,
    items: list[dict],
    keywords: list[str],
) -> list[dict]:
    """Return deep-copied items with rewritten bullet_points."""
    out = copy.deepcopy(items)
    for item in out:
        context = item.get("title") or item.get("name") or ""
        if item.get("company"):
            context += f" at {item['company']}"
        tasks = [
            _rewrite_bullet(client, b, keywords, context)
            for b in item.get("bullet_points", [])
        ]
        item["bullet_points"] = list(await asyncio.gather(*tasks))
    return out


def _render_pdf(resume_data: dict, template_path: str) -> bytes:
    """Render resume_data through the Jinja2 HTML template → PDF bytes."""
    template_dir  = os.path.dirname(os.path.abspath(template_path))
    template_name = os.path.basename(template_path)
    env      = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)
    html_out = template.render(**resume_data)

    buf = io.BytesIO()
    result = pisa.CreatePDF(html_out, dest=buf)
    if result.err:
        raise RuntimeError(f"xhtml2pdf error code {result.err}")
    return buf.getvalue()


# ════════════════════════════════════════════════════════════════════════════
# POST /resume/tailor
# ════════════════════════════════════════════════════════════════════════════

@app.post("/resume/tailor")
async def tailor_resume(
    job_id: str = Query(..., description="The job_id string from job_search_results"),
    user_id: str = Depends(get_current_user),
):
    """
    1. Load profile (experience_raw, projects_raw, education, skills, …)
    2. Load job from job_search_results by job_id
    3. AI-rewrite bullets toward the job description
    4. Render HTML template → PDF
    5. Save tailored data + PDF to tailored_resumes collection
    6. Return PDF as application/pdf download
    """
    # ── 1. Load profile ──────────────────────────────────────────────────────
    profile = await mongo.get_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    # ── 2. Load job ───────────────────────────────────────────────────────────
    job = await mongo.get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    job_description: str = job.get("description", "")
    job_title: str       = job.get("title", "")
    job_company: str     = job.get("company", "")

    # ── 3. AI rewrite ─────────────────────────────────────────────────────────
    openai_client = get_async_openai_client()

    keywords = await _extract_keywords(openai_client, job_description)

    experience_rewritten = await _rewrite_all_bullets(
        openai_client,
        profile.get("experience_raw") or [],
        keywords,
    )
    projects_rewritten = await _rewrite_all_bullets(
        openai_client,
        profile.get("projects_raw") or [],
        keywords,
    )

    # ── 4. Build resume_data dict for Jinja2 template ─────────────────────────
    resume_data = {
        "name":               profile.get("name", ""),
        "email":              profile.get("email", ""),
        "phone":              profile.get("phone", ""),
        "linkedin":           profile.get("linkedin", ""),
        "github":             profile.get("github", ""),
        "location":           profile.get("location", ""),
        "summary":            "",                              # hidden unless GENERATE_SUMMARY=true
        "education":          profile.get("education", []),
        "skills":             profile.get("skills", []),
        "categorized_skills": _categorise_skills(profile.get("skills", [])),
        "certifications":     profile.get("certifications", []),
        "achievements":       profile.get("achievements", []),
        "experience":         experience_rewritten,
        "projects":           projects_rewritten,
    }

    # ── 5. Render PDF ─────────────────────────────────────────────────────────
    template_path = os.path.join(os.path.dirname(__file__), "resume_template.html")
    try:
        pdf_bytes = await asyncio.get_event_loop().run_in_executor(
            None, _render_pdf, resume_data, template_path
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"PDF rendering failed: {e}")

    # ── 6. Save to MongoDB ────────────────────────────────────────────────────
    await mongo.save_tailored_resume(
        user_id=user_id,
        job_id=job_id,
        job_title=job_title,
        job_company=job_company,
        tailored_data=resume_data,
        pdf_bytes=pdf_bytes,
    )

    # ── 7. Return PDF ─────────────────────────────────────────────────────────
    safe_company = re.sub(r"[^a-zA-Z0-9_-]", "-", job_company).strip("-").lower()
    filename = f"resume-{safe_company}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ════════════════════════════════════════════════════════════════════════════
# GET /resume/tailor  (re-download without re-running AI)
# ════════════════════════════════════════════════════════════════════════════

@app.get("/resume/tailor")
async def download_tailored_resume(
    job_id: str = Query(...),
    user_id: str = Depends(get_current_user),
):
    """Return the previously generated tailored PDF without re-running AI."""
    doc = await mongo.get_tailored_resume(user_id, job_id)
    if not doc:
        raise HTTPException(status_code=404, detail="No tailored resume found for this job. Generate one first.")

    pdf_bytes   = doc.get("pdf_bytes", b"")
    job_company = doc.get("job_company", "job")
    safe_company = re.sub(r"[^a-zA-Z0-9_-]", "-", job_company).strip("-").lower()

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="resume-{safe_company}.pdf"'},
    )


# ---- ORCHESTRATOR ROUTE ----

@app.post("/query")
async def query(body: QueryRequest, user_id: str = Depends(get_current_user)):
    student_profile = await mongo.get_profile(user_id)
    if not student_profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    history = body.history if body.history else await memory.get_history(user_id)

    try:
        result = await run_orchestrator(user_id, body.query, student_profile, history)
    except Exception as e:
        raise HTTPException(status_code=504, detail=str(e))

    answer_text = result.get("answer") or result.get("results") or result.get("response") or ""
    await memory.add_message(user_id, "human", body.query)
    await memory.add_message(user_id, "ai", str(answer_text))

    return {
        "query": body.query,
        **result
    }


# ---- HISTORY ROUTES ----

@app.get("/history")
async def get_history(user_id: str = Depends(get_current_user)):
    history = await memory.get_history(user_id)
    return {"history": history}

@app.delete("/history")
async def clear_history(user_id: str = Depends(get_current_user)):
    await memory.clear_history(user_id)
    return {"message": "History cleared"}


# ---- HEALTH CHECK ----

@app.get("/health")
async def health():
    return {"status": "ok"}