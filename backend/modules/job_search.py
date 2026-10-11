#Import statements.
import os
import math
import re
import certifi
import requests
import json
from llm import get_openai_client, model_name
from collections import Counter
from typing import Dict, Any, List

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

#
def run_module_3_sync():
    aid = os.getenv("ADZUNA_APP_ID")
    akey = os.getenv("ADZUNA_API_KEY")

    url = f"https://api.adzuna.com/v1/api/jobs/us/search/1?app_id={aid}&app_key={akey}&what=engineer"
    response = requests.get(url)

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
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = response.choices[0].message.content
        data = json.loads(content)
        return [str(s).lower().strip() for s in data.get("skills", [])][:8]

    except Exception as e:
        print("OpenAI skill extraction failed:", e)
        return []

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


def derive_keywords(profile: Dict[str, Any]) -> str:
    parts = (
        profile.get("skills", [])
        + profile.get("coursework", [])
        + [profile.get("major", ""), profile.get("target_role", "")]
    )
    return " ".join(str(x).strip() for x in parts if str(x).strip())


def fetch_kaggle_jobs(profile: Dict[str, Any]):
    jobs = []
    # Pulling all 1167 rows from your dataset
    docs = list(db["all_job_kaggle"].find())

    for doc in docs:
        # Standardizing keys to match your MongoDB row
        jobs.append({
            "job_id": doc.get("job_id"),
            "title": doc.get("job_title", "Unknown"),
            "company": doc.get("category", "Industry Professional"), 
            "location": "Dallas, TX",
            "description": doc.get("job_description", ""),
            "skills": doc.get("job_skill_set", []),
            "url": f"https://jobs.example.com/{doc.get('job_id')}",
            "salary": None,
            "source": "all_job_kaggle"
        })
    return jobs
def fetch_adzuna_jobs(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    aid = os.getenv("ADZUNA_APP_ID")
    akey = os.getenv("ADZUNA_API_KEY")

    if not aid or not akey:
        print("Missing Adzuna credentials in .env")
        return []

    keyword = derive_keywords(profile) or "engineer"

    params = {
        "app_id": aid,
        "app_key": akey,
        "what": keyword,
        "results_per_page": 50
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
            jobs.append({
                "job_id": doc.get("id"),
                "title": doc.get("title", "Unknown"),
                "company": doc.get("company", {}).get("display_name", "Unknown Company"),
                "location": doc.get("location", {}).get("display_name", "Unknown Location"),
                "description": doc.get("description", ""),
                "skills": [],
                "url": doc.get("redirect_url", ""),
                "salary": None,
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

    keyword = str(profile.get("major", "")).strip() or "engineer"

    headers = {
        "Host": "data.usajobs.gov",
        "User-Agent": user_agent,
        "Authorization-Key": api_key
    }

    params = {
        "Keyword": keyword,
        "ResultsPerPage": 50
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

            jobs.append({
                "job_id": desc.get("PositionID"),
                "title": desc.get("PositionTitle", "Unknown"),
                "company": desc.get("OrganizationName", "USAJobs"),
                "location": desc.get("PositionLocationDisplay", "Unknown"),
                "description": details.get("JobSummary", ""),
                "skills": [],
                "url": apply_uri[0] if apply_uri else "",
                "salary": None,
                "source": "usajobs",
                "date_posted": desc.get("PublicationStartDate", "N/A")
            })
        return jobs

    except requests.exceptions.RequestException as e:
        print(f"USAJobs request failed: {e}")
        return []

def fetch_all_jobs(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    #kaggle_jobs = fetch_kaggle_jobs(profile)
    adzuna_jobs = fetch_adzuna_jobs(profile)
    usajobs_jobs = fetch_usajobs_jobs(profile)

    #print("Kaggle jobs fetched:", len(kaggle_jobs))
    print("Adzuna jobs fetched:", len(adzuna_jobs))
    #print("USAJobs jobs fetched:", len(usajobs_jobs))
    print("Mongo URI:", MONGODB_URI)
    jobs = []
    #jobs.extend(kaggle_jobs)
    jobs.extend(adzuna_jobs)
    jobs.extend(usajobs_jobs)

    return jobs

def score_job(job, profile):
    p_skills = [s.lower().strip() for s in profile.get("skills", [])]
    preferred_locations = [loc.lower().strip() for loc in profile.get("location_preference", [])]
    target_role = str(profile.get("target_role", "")).lower().strip()
    major = str(profile.get("major", "")).lower().strip()

    title = str(job.get("title", "")).lower()
    description = str(job.get("description", "")).lower()
    location = str(job.get("location", "")).lower()
    job_text = f"{title} {description}"
    job_skill_list = [str(s).lower().strip() for s in job.get("skills", [])]

    matched = [s for s in p_skills if s in job_text or s in job_skill_list]

    overlap = len(matched) / max(len(p_skills), 1)
    sim = cosine_similarity(tokenize(" ".join(p_skills + [major])), tokenize(job_text))

    title_score = 0.0
    if target_role and target_role in title:
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

    final_score = (
        0.45 * overlap +
        0.25 * sim +
        0.20 * title_score +
        0.10 * location_score
    )

    job["match_score"] = round(final_score, 4)
    job["matched_skills"] = matched
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


def run_job_search(profile: Dict[str, Any]) -> Dict[str, Any]:
    all_jobs = fetch_all_jobs(profile)

    for job in all_jobs:
        score_job(job, profile)

    sorted_jobs = sorted(all_jobs, key=lambda x: x.get("match_score", 0), reverse=True)

    #for job in sorted_jobs[:15]:
        #print(job["source"], job["title"], job["match_score"])
    best_adzuna = next((job for job in sorted_jobs if job["source"] == "adzuna"), None)
    best_usajobs = next((job for job in sorted_jobs if job["source"] == "usajobs"), None)

    
    top_adzuna = [job for job in sorted_jobs if job["source"] == "adzuna"][:2]
    top_usajobs = [job for job in sorted_jobs if job["source"] == "usajobs"][:5]
    #top_kaggle = [job for job in sorted_jobs if job["source"] == "all_job_kaggle"][:1]

    ranked = top_adzuna + top_usajobs #+ top_kaggle
    if len(ranked) < 5:
        used_ids = {job["job_id"] for job in ranked}
        fillers = [job for job in sorted_jobs if job["job_id"] not in used_ids]
        ranked.extend(fillers[:5 - len(ranked)])

    ranked = ranked[:5]
    for job in ranked:
        job["skills"] = extract_job_skills_openai( job.get("title", ""), job.get("description", "") )
        profile_skills = [s.lower().strip() for s in profile.get("skills", [])]
        job_skills = [s.lower().strip() for s in job.get("skills", [])]
        job_skills_text = " ".join(job_skills)
        job["matched_skills"] = [ ps for ps in profile_skills if any(ps in js or js in ps for js in job_skills) ]
        if not job.get("salary"):
            job["salary"] = enrich_salary(job["title"])["median"]

    benchmark = enrich_salary(ranked[0]["title"]) if ranked else {"median": "N/A", "growth_rate": "N/A"}

    return {
        "jobs": ranked,
        "salary_benchmark": benchmark,
        "sources": {
            "job_sources": list({job["source"] for job in ranked}),
            "salary_source": "salaries"
        }
    }

# For quick testing
if __name__ == "__main__":
    test_profile = {
          "skills": ["python", "java", "sql", "git"],
          "major": "Computer Science",
          "coursework": ["software engineering", "Database Design", "Virtual Reality"],
          "location_preference": ["Dallas", "Remote"],
          "target_role": "entry"
    }
    print("OPENAI KEY LOADED:", bool(os.getenv("OPENAI_API_KEY")))

    print(extract_job_skills_openai(
        "Data Analyst",
        "Analyze datasets using SQL, Excel, Tableau, Python, statistics, and communicate insights."
    ))

    result = run_job_search(test_profile)

    from pprint import pprint
    pprint(result)
    