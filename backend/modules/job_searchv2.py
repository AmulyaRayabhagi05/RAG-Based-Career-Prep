#Import statements.
import os
import math
import re
import certifi
import requests
import json
import time
from datetime import datetime, timezone, timedelta
from llm import get_openai_client, model_name
from collections import Counter
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
openai_client = get_openai_client()
#Reads Mongo thing.
MONGODB_URI = os.getenv("MONGO_URI")
MONGODB_DB = os.getenv("DB_NAME", "elevatr_db")

ca = certifi.where()
client = MongoClient(MONGODB_URI, tlsCAFile=ca)
db = client[MONGODB_DB]

# --- Job cache helpers (cache fetched API results to speed subsequent searches) ---


def ensure_job_cache_indexes():
    try:
        # unique index for id per source, keywords for fast lookups by derived terms, and fetched_at for TTL-style queries
        db.job_cache.create_index([("source", 1), ("job_id", 1)], unique=True)
        db.job_cache.create_index([("keywords", 1)])
        db.job_cache.create_index([("fetched_at", 1)])
    except Exception as e:
        print("ensure_job_cache_indexes failed:", e)

# Make sure indexes exist at import time
ensure_job_cache_indexes()


def job_cache_upsert(job: Dict[str, Any], derived_keyword: str = None):
    """Upsert a normalized job record into job_cache. Stores a 'raw' copy for compatibility."""
    try:
        title = job.get("title", "") or ""
        description = job.get("description", "") or ""
        tokens = tokenize(f"{title} {description}")
        keywords = list(set(tokens + (tokenize(derived_keyword) if derived_keyword else [])))
        record = {
            "source": job.get("source"),
            "job_id": job.get("job_id"),
            "title": title,
            "company": job.get("company"),
            "location": job.get("location"),
            "description": description,
            "skills": job.get("skills", []),
            "url": job.get("url"),
            "salary": job.get("salary"),
            "date_posted": job.get("date_posted"),
            "keywords": keywords,
            "fetched_at": datetime.now(timezone.utc),
            "raw": job
        }
        db.job_cache.update_one(
            {"source": record["source"], "job_id": record["job_id"]},
            {"$set": record},
            upsert=True
        )
    except Exception as e:
        print("job_cache_upsert failed:", e)

#
def run_module_3_sync():
    aid = os.getenv("ADZUNA_APP_ID")
    akey = os.getenv("ADZUNA_API_KEY")
    print("ADZUNA_APP_ID loaded:", bool(aid))
    print("ADZUNA_API_KEY loaded:", bool(akey))
    print("Adzuna keyword:", keyword)
    url = f"https://api.adzuna.com/v1/api/jobs/us/search/1?app_id={aid}&app_key={akey}&what=engineer"
    response = requests.get(url)
    print("Adzuna status code:", response.status_code)
    print("Adzuna response preview:", response.text[:300])
    if response.status_code == 200:
        data = response.json()
        jobs = data.get("results", [])

        if jobs:
            db.all_job_posts.delete_many({})
            db.all_job_posts.insert_many(jobs)
            print(f"Module 3: {len(jobs)} live jobs synced to Mongo Atlas.")
    else:
        print(f"Error {response.status_code}: Check your .env keys.")


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[a-zA-Z0-9\+\#\.]+", text.lower())

SKILL_VOCABULARY = [
    "python", "java", "javascript", "typescript", "react", "node.js", "sql",
    "mysql", "postgresql", "mongodb", "aws", "azure", "docker", "kubernetes",
    "git", "linux", "html", "css", "fastapi", "flask", "spring boot",
    "excel", "tableau", "power bi", "statistics", "data analysis",
    "machine learning", "tensorflow", "pytorch",
    "human resources", "recruiting", "recruitment", "onboarding",
    "payroll", "benefits", "employee relations", "talent acquisition",
    "hris", "communication", "leadership", "project management",
    "customer service", "sales", "marketing", "research", "writing",
    "presentation", "training", "documentation"
]

def extract_job_skills(title: str, description: str) -> List[str]:
    text = f"{title} {description}".lower()
    found = []

    for skill in SKILL_VOCABULARY:
        if re.search(rf"\b{re.escape(skill)}\b", text):
            found.append(skill)

    return found[:8]

def cosine_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    c1 = Counter(tokens1)
    c2 = Counter(tokens2)

    common = set(c1) & set(c2)
    dot = sum(c1[t] * c2[t] for t in common)

    mag1 = math.sqrt(sum(v * v for v in c1.values()))
    mag2 = math.sqrt(sum(v * v for v in c2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


ROLE_FOCUS_PATTERNS = [
    ("software engineer", [
        r"software engineer", r"software development", r"backend", r"frontend", r"full[- ]stack",
        r"web developer", r"application developer", r"system design", r"distributed systems",
        r"react", r"typescript", r"javascript", r"node\.js", r"api", r"engineering"
    ]),
    ("data scientist", [r"data scientist", r"data analysis", r"statistics", r"machine learning", r"ml", r"pandas", r"tableau", r"power bi"]),
    ("machine learning engineer", [r"machine learning engineer", r"ml engineer", r"machine learning", r"deep learning", r"pytorch", r"tensorflow", r"nlp"]),
    ("devops engineer", [r"devops", r"site reliability", r"sre", r"kubernetes", r"docker", r"terraform", r"aws", r"azure", r"cloud"]),
    ("product manager", [r"product manager", r"product management", r"roadmap", r"stakeholder", r"product strategy"]),
]


def infer_role_focus(profile: Dict[str, Any]) -> str:
    parts = [
        str(profile.get("query", "")),
        str(profile.get("target_role", "")),
        str(profile.get("resumeText", "")),
        str(profile.get("current_role", "")),
        str(profile.get("current_org", "")),
        str(profile.get("major", "")),
        " ".join(profile.get("skills", []) or []),
        " ".join(profile.get("coursework", []) or []),
    ]
    text = " ".join(parts).lower()

    for role, patterns in ROLE_FOCUS_PATTERNS:
        if any(re.search(pattern, text) for pattern in patterns):
            return role

    if str(profile.get("target_role", "")).strip():
        return str(profile.get("target_role", "")).strip()

    if str(profile.get("major", "")).strip():
        return str(profile.get("major", "")).strip()

    return "software engineer"


def derive_keywords(profile: Dict[str, Any]) -> str:
    query = str(profile.get("query", "")).strip()
    inferred_role = infer_role_focus(profile)

    def looks_like_role(text: str) -> bool:
        return any(re.search(pattern, text.lower()) for _, patterns in ROLE_FOCUS_PATTERNS for pattern in patterns)

    if query and looks_like_role(query):
        return query

    return inferred_role


def format_salary(min_salary=None, max_salary=None):
    try:
        if min_salary and max_salary:
            return f"${int(min_salary):,} - ${int(max_salary):,}"
        if min_salary:
            return f"${int(min_salary):,}+"
        if max_salary:
            return f"Up to ${int(max_salary):,}"
    except:
        pass
    return None

def fetch_adzuna_jobs(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    aid = os.getenv("ADZUNA_APP_ID")
    akey = os.getenv("ADZUNA_API_KEY")

    if not aid or not akey:
        print("Missing Adzuna credentials in .env")
        return []

    keyword = derive_keywords(profile) #or "software engineer"
    location = str(profile.get("location", "")).strip()
    if not location:
        preferred_locations = profile.get("location_preference", []) or []
        location = preferred_locations[0] if preferred_locations else ""

    params = {
        "app_id": aid,
        "app_key": akey,
        "what": keyword,
        "results_per_page": 50,
        "where": location or None,
    }

    try:
        response = requests.get(
            "https://api.adzuna.com/v1/api/jobs/us/search/1",
            params=params,
            timeout=20
        )
        

        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])

        jobs = []
        for doc in results:
            title = doc.get("title", "Unknown")
            salary_min = doc.get("salary_min")
            salary_max = doc.get("salary_max")
            salary_match = re.search(
                 r"\$\d+(?:\.\d+)?(?:\s*-\s*\$?\d+(?:\.\d+)?)?(?:/hr| per hour| hourly)?",
                title
            )
            title_salary = salary_match.group() if salary_match else None
            def fetch_full_description_from_url(url: str) -> str:
                if not url:
                    return ""

                try:
                    res = requests.get(
                        url,
                        timeout=10,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )

                    text = re.sub(r"<script.*?</script>", "", res.text, flags=re.DOTALL)
                    text = re.sub(r"<style.*?</style>", "", text, flags=re.DOTALL)
                    text = re.sub(r"<[^>]+>", " ", text)
                    text = re.sub(r"\s+", " ", text).strip()

                # cut off junk after apply section
                    cut_phrases = [
                        "Apply now",
                        "Apply Now",
                        "Apply for this job",
                        "Apply on company site",
                        "Create job alert",
                        "Similar jobs",
                    ]

                    for phrase in cut_phrases:
                        idx = text.find(phrase)
                        if idx != -1:
                            text = text[:idx].strip()
                            break

                    return text[:5000]

                except Exception as e:
                    print("Full desc fetch failed:", e)
                    return ""
            desc_text = re.sub(r"<[^>]+>", "", doc.get("description", "") or "")
            jobs.append({
                "job_id": doc.get("id"),
                "title": doc.get("title", "Unknown"),
                "company": doc.get("company", {}).get("display_name", "Unknown Company"),
                "location": doc.get("location", {}).get("display_name", "Unknown Location"),
                "description": desc_text,
                "skills": [],
                "url": doc.get("redirect_url", ""),
                "salary": format_salary(salary_min, salary_max) or title_salary,
                "source": "adzuna",
                "date_posted": doc.get("created", "N/A")
            })
        return jobs

    except requests.exceptions.RequestException as e:
        print(f"Adzuna request failed: {e}")
        return []
def fetch_usajobs_jobs(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    api_key = os.getenv("USAJOBS_API_KEY")
    user_agent = os.getenv("USAJOBS_USER_AGENT")

    if not api_key or not user_agent:
        print("Missing USAJOBS credentials in .env")
        return []

    keyword = derive_keywords(profile)
    location = str(profile.get("location", "")).strip()
    if not location:
        preferred_locations = profile.get("location_preference", []) or []
        location = preferred_locations[0] if preferred_locations else ""

    headers = {
        "Host": "data.usajobs.gov",
        "User-Agent": user_agent,
        "Authorization-Key": api_key
    }

    params = {
        "Keyword": keyword,
        "ResultsPerPage": 50,
        "LocationName": location or None
    }

    try:
        response = requests.get(
            "https://data.usajobs.gov/api/search",
            headers=headers,
            params=params,
            timeout=20
        )
        

        response.raise_for_status()
        data = response.json()

        items = data.get("SearchResult", {}).get("SearchResultItems", [])
        jobs = []

        for item in items:
            desc = item.get("MatchedObjectDescriptor", {})
            details = desc.get("UserArea", {}).get("Details", {})
            apply_uri = desc.get("ApplyURI", [])
            remuneration = desc.get("PositionRemuneration", [])
            salary_info = remuneration[0] if remuneration else {}

            salary = format_salary(
                salary_info.get("MinimumRange"),
                salary_info.get("MaximumRange")
            )
            def clean_field(value):
                if isinstance(value, list):
                    return " ".join(str(item) for item in value if item)
                if value is None:
                    return ""
                return str(value)

            description = " ".join(filter(None, [
                    clean_field(details.get("JobSummary")),
                    clean_field(details.get("MajorDuties")),
                    clean_field(details.get("Requirements")),
                    clean_field(details.get("Education")),
                ])).strip()

            description = description.split("Additional Information")[0].strip()
            description = re.sub(r"\s+", " ", description)
            jobs.append({
                "job_id": desc.get("PositionID"),
                "title": desc.get("PositionTitle", "Unknown"),
                "company": desc.get("OrganizationName", "USAJobs"),
                "location": desc.get("PositionLocationDisplay", "Unknown"),
                "description": description,
                "skills": [],
                "url": apply_uri[0] if apply_uri else "",
                "salary": salary,
                "source": "usajobs",
                "date_posted": desc.get("PublicationStartDate", "N/A")
                })
        return jobs

    except requests.exceptions.RequestException as e:
        print(f"USAJobs request failed: {e}")
        return []

def fetch_all_jobs(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Cache-first job retrieval. Looks for recent cached jobs matching tokens from the derived keyword,
    otherwise fetches from configured sources in parallel and upserts results into the cache for future queries.
    """
    keyword = derive_keywords(profile).lower()
    kw_tokens = [t for t in tokenize(keyword) if t]
    ttl_seconds = int(os.getenv("JOB_CACHE_TTL_SECONDS", "3600"))
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)

    try:
        query = {"fetched_at": {"$gte": cutoff}}
        if kw_tokens:
            query["keywords"] = {"$in": kw_tokens}

        cached_cursor = db.job_cache.find(query).limit(200)
        cached = list(cached_cursor)
        if cached:
            print(f"job_cache: cache hit, {len(cached)} jobs")
            jobs = []
            for doc in cached:
                if isinstance(doc.get("raw"), dict):
                    jobs.append(doc.get("raw"))
                else:
                    jobs.append({
                        "job_id": doc.get("job_id"),
                        "title": doc.get("title"),
                        "company": doc.get("company"),
                        "location": doc.get("location"),
                        "description": doc.get("description"),
                        "skills": doc.get("skills", []),
                        "url": doc.get("url"),
                        "salary": doc.get("salary"),
                        "source": doc.get("source"),
                        "date_posted": doc.get("date_posted")
                    })
            return jobs
        else:
            print("job_cache: cache miss")
    except Exception as e:
        print("job_cache lookup failed:", e)

    # Cache miss: fetch from sources in parallel
    jobs = []
    sources = [fetch_usajobs_jobs, fetch_adzuna_jobs]
    try:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=len(sources)) as ex:
            futures = {ex.submit(fn, profile): fn.__name__ for fn in sources}
            for fut in as_completed(futures):
                try:
                    src_jobs = fut.result()
                    if src_jobs:
                        jobs.extend(src_jobs)
                except Exception as e:
                    print("source fetch failed:", e)
        print(f"parallel fetch took {time.time() - t0:.2f}s")
    except Exception as e:
        print("parallel fetch failed:", e)

    # Upsert fetched jobs into cache
    for job in jobs:
        try:
            job_cache_upsert(job, keyword)
        except Exception as e:
            print("job_cache_upsert during fetch_all_jobs failed:", e)

    return jobs


def is_workable_url(url: str, source: str = "") -> bool:
    if not url or not isinstance(url, str):
        return False

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return False

    if source.lower() == "adzuna":
        try:
            response = requests.get(
                url,
                timeout=8,
                allow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            return 200 <= response.status_code < 400
        except Exception:
            return False

    return True

def score_job(job, profile):
    p_skills = [s.lower().strip() for s in profile.get("skills", [])]
    preferred_locations = [loc.lower().strip() for loc in profile.get("location_preference", [])]
    target_role = str(profile.get("target_role", "")).lower().strip()
    major = str(profile.get("major", "")).lower().strip()
    query = str(profile.get("query", "")).lower().strip()
    source_filters = {str(s).lower().strip() for s in profile.get("sources", []) if str(s).strip()}
    company_filter = str(profile.get("company", "")).lower().strip()
    location_filter = str(profile.get("location", "")).lower().strip()
    remote_only = bool(profile.get("remote_only", False))
    posted_within_days = int(profile.get("posted_within_days", 0) or 0)

    title = str(job.get("title", "")).lower()
    description = str(job.get("description", "")).lower()
    location = str(job.get("location", "")).lower()
    source = str(job.get("source", "")).lower().strip()
    job_text = f"{title} {description}"
    required_skills = extract_job_skills(title, description)
    job_skill_list = [str(s).lower().strip() for s in job.get("skills", [])]

    matched = [s for s in p_skills if s in job_text or s in job_skill_list]

    overlap = len(matched) / max(len(p_skills), 1)
    sim = cosine_similarity(tokenize(" ".join(p_skills + [major])), tokenize(job_text))

    title_score = 0.0
    if query and query in job_text:
        title_score = 1.0
    elif target_role and target_role in title:
        title_score = 1.0
    elif major and major in title:
        title_score = 0.7
    elif major and major in description:
        title_score = 0.4

    location_score = 0.0
    if any(loc in location for loc in preferred_locations):
        location_score = 1.0
    elif "remote" in preferred_locations and "remote" in location:
        location_score = 1.0
    elif location_filter and location_filter in location:
        location_score = 1.0

    source_score = 0.0
    if source_filters:
        source_score = 1.0 if source in source_filters else -0.5

    company_score = 0.0
    if company_filter:
        company_score = 1.0 if company_filter in str(job.get("company", "")).lower() else -0.25

    recency_score = 0.0
    date_posted = job.get("date_posted")
    if posted_within_days and date_posted:
        try:
            days_old = (datetime.now(timezone.utc) - datetime.fromisoformat(str(date_posted).replace("Z", "+00:00"))).days
            recency_score = 1.0 if days_old <= posted_within_days else -0.5
        except Exception:
            recency_score = 0.0

    if remote_only and "remote" not in location:
        job["match_score"] = 0.0
        job["matched_skills"] = matched
        job["skills"] = required_skills
        return 0.0

    final_score = (
        0.45 * overlap +
        0.25 * sim +
        0.20 * title_score +
        0.08 * location_score +
        0.04 * source_score +
        0.02 * company_score +
        0.01 * recency_score
    )

    job["match_score"] = round(final_score, 4)
    job["matched_skills"] = matched
    job["skills"] = required_skills
    return final_score


def enrich_salary(title: str) -> Dict[str, Any]:
    # Corrected key 'job_title' to match your 'Accountant' data
    return {"median": "N/A", "growth_rate": "N/A"}
    if salary_doc:
        # Pulling from 'salary_annual' and 'job_growth_rate_percent'
        median = salary_doc.get("salary_annual") or salary_doc.get("median_pay", {}).get("annual")
        growth = salary_doc.get("job_growth_rate_percent")
        
        return {
            "median": f"${median:,}" if median else "N/A",
            "growth_rate": f"{growth}%" if growth else "N/A"
        }
    return {"median": "N/A", "growth_rate": "N/A"}


def build_timeline(title: str) -> List[Dict[str, Any]]:
    timeline_doc = db["job_future"].find_one({
        "$or": [
            {"family": {"$regex": re.escape(title), "$options": "i"}},
            {"title": {"$regex": re.escape(title), "$options": "i"}},
            {"role": {"$regex": re.escape(title), "$options": "i"}}
        ]
    })

    if timeline_doc and "timeline" in timeline_doc:
        return timeline_doc["timeline"]

    return [
        {
            "stage": "entry",
            "years": "0-2",
            "roles": [f"Junior {title}", "Intern"],
            "salary": "See salary benchmark"
        },
        {
            "stage": "mid",
            "years": "3-5",
            "roles": [title, f"Mid-level {title}"],
            "salary": "See salary benchmark"
        },
        {
            "stage": "senior",
            "years": "6+",
            "roles": [f"Senior {title}", f"Lead {title}"],
            "salary": "See salary benchmark"
        }
    ]

def extract_job_skills_openai(title: str, description: str) -> List[str]:
    if not os.getenv("OPENAI_API_KEY"):
        print("No OpenAI key found")
        return []

    prompt = f"""
    Extract 5 to 8 concrete skills required for this job.
    Only return valid JSON like:
    {{"skills": ["skill 1", "skill 2"]}}

    Title: {title}
    Description: {description[:2000]}
    """

    try:
        response = openai_client.chat.completions.create(
            model=model_name("small"),
            messages=[ {"role": "system", "content": "You extract job skills. Always return valid JSON only."}, 
                       {"role": "user", "content": prompt} ],
            temperature=0, 
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        data = json.loads(content)
        return [str(s).lower().strip() for s in data.get("skills", [])][:8]

    except Exception as e:
        print("OpenAI skill extraction failed:", e)
        return []

def normalize_skill(skill: str) -> str:
    return skill.lower().strip()

def skill_matches(user_skill: str, job_skill: str) -> bool:
    user_skill = normalize_skill(user_skill)
    job_skill = normalize_skill(job_skill)

    return (
        user_skill == job_skill
        or user_skill in job_skill
        or job_skill in user_skill
    )
def run_job_search(profile: Dict[str, Any]) -> Dict[str, Any]:
    # Fetch and score
    all_jobs = fetch_all_jobs(profile)

    if profile.get("sources"):
        allowed_sources = {str(s).lower().strip() for s in profile.get("sources", []) if str(s).strip()}
        all_jobs = [job for job in all_jobs if str(job.get("source", "")).lower().strip() in allowed_sources]

    for job in all_jobs:
        score_job(job, profile)

    sorted_jobs = sorted(all_jobs, key=lambda x: x.get("match_score", 0), reverse=True)

    ranked = []
    seen_ids = set()
    allowed_sources = [str(s).lower().strip() for s in profile.get("sources", []) if str(s).strip()]
    source_priority = ["usajobs", "adzuna"]
    if allowed_sources:
        source_priority = [s for s in source_priority if s in allowed_sources] or allowed_sources

    quotas = {"usajobs": 5, "adzuna": 1}

    for source_name in source_priority:
        quota = quotas.get(source_name, 1)
        picked_for_source = 0
        for job in sorted_jobs:
            if str(job.get("source", "")).lower() != source_name:
                continue
            if picked_for_source >= quota:
                break
            job_id = job.get("job_id")
            if job_id in seen_ids:
                continue
            if not is_workable_url(job.get("url", ""), source_name):
                continue
            ranked.append(job)
            seen_ids.add(job_id)
            picked_for_source += 1
            if len(ranked) >= 6:
                break
        if len(ranked) >= 6:
            break

    if len(ranked) < 6:
        for job in sorted_jobs:
            job_id = job.get("job_id")
            if job_id in seen_ids:
                continue
            source_name = str(job.get("source", "")).lower()
            if allowed_sources and source_name not in allowed_sources:
                continue
            if not is_workable_url(job.get("url", ""), source_name):
                continue
            ranked.append(job)
            seen_ids.add(job_id)
            if len(ranked) >= 6:
                break

    ranked = ranked[:6]

    # Avoid expensive OpenAI calls by default. Enable with JOB_USE_OPENAI_SKILLS=1.
    use_openai = os.getenv("JOB_USE_OPENAI_SKILLS", "0") == "1" and os.getenv("OPENAI_API_KEY")
    derived_kw = derive_keywords(profile)

    for job in ranked:
        # Prefer already-present or heuristic skills
        skills = job.get("skills") or []
        if not skills:
            skills = extract_job_skills(job.get("title", ""), job.get("description", ""))
            job["skills"] = skills

        # Optionally use OpenAI for higher-quality extraction when explicitly enabled
        if use_openai and not job.get("skills"):
            try:
                openai_skills = extract_job_skills_openai(job.get("title", ""), job.get("description", ""))
                if openai_skills:
                    job["skills"] = openai_skills
            except Exception as e:
                print("OpenAI skill extraction failed:", e)

        profile_skills = profile.get("skills", [])
        matched = []
        for user_skill in profile_skills:
            for job_skill in job.get("skills", []):
                if skill_matches(user_skill, job_skill):
                    matched.append(job_skill)
                    break
        job["matched_skills"] = matched

        if not job.get("salary"):
            job["salary"] = enrich_salary(job.get("title", ""))["median"]

        # Persist enriched job back to cache to avoid re-querying OpenAI on subsequent runs
        try:
            job_cache_upsert(job, derived_kw)
        except Exception as e:
            print("job_cache upsert during enrichment failed:", e)

    benchmark = enrich_salary(ranked[0].get("title")) if ranked else {"median": "N/A", "growth_rate": "N/A"}
    sources = {"job_sources": list({job.get("source") for job in ranked}), "salary_source": "salaries"}

    if ranked:
        try:
            db.job_search_results.delete_many({"profile": profile})
            db.job_search_results.insert_one({
                "profile": profile,
                "jobs": ranked,
                "salary_benchmark": benchmark,
                "sources": sources,
                "created_at": datetime.now(timezone.utc)
            })
        except Exception as e:
            print("saving job_search_results failed:", e)

    return {
        "jobs": ranked,
        "salary_benchmark": benchmark,
        "sources": sources
    }

# For quick testing
if __name__ == "__main__":
    test_profile = {
          "skills": ["machine learning", "data visualization", "data cleaning", "research", "project management", "software engineering"],
          "major": "Computer Science",
          "coursework": ["Data Structures"],
          "location_preference": ["Dallas", "Remote"],
          "target_role": "entry"
    }

    result = run_job_search(test_profile)
    print("FINAL JOB COUNT:", len(result["jobs"]))
    from pprint import pprint
    pprint(result)
