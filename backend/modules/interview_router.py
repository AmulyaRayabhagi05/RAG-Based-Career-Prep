from .interview import generate_questions, generate_feedback
from fastapi import APIRouter
from pydantic import BaseModel
import os
import pandas as pd
from llm import get_openai_client
import requests
import json
from dotenv import load_dotenv
from rag.retriever import retrieve_for_interview_cached

load_dotenv()

router = APIRouter(prefix="/interview", tags=["interview"])
client = get_openai_client()

class AnswerRequest(BaseModel):
    question: str
    answer: str

class QuestionRequest(BaseModel):
    selection: str  # "HR" or "Technical"
    target_job: str = "Financial Advisor"
    first_question: bool = False


#pd.read_csv("backend/modules/new_behavioral_interview_questions_dataset.csv")
#pd.read_csv("backend/modules/Task Statements.csv")

@router.post("/question")
def get_question(body: QuestionRequest):

    if body.first_question:
        return {"question": "Tell me about yourself"}

    rag_results = retrieve_for_interview_cached(body.target_job)

    if body.selection == "HR":
        csv_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "new_behavioral_interview_questions_dataset.csv"))
        
        sample = csv_data.sample(10).iloc[:, 0].tolist()
        
        combined_questions = sample + rag_results["questions"]
        
        question = generate_questions(body.selection, combined_questions, body.target_job, client)

    else:
        csv_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "Task Statements.csv"))
        csv_tasks = csv_data[csv_data['Title'].str.contains(body.target_job, case=False, na=False)]['Task'].tolist()[:10]
        
        combined_tasks = csv_tasks + rag_results["tasks"]
        
        question = generate_questions(body.selection, combined_tasks, body.target_job, client)

    return {"question": question}

@router.post("/feedback")
def get_feedback(body: AnswerRequest):
    result = generate_feedback(body.question, body.answer, client)
    return json.loads(result)

