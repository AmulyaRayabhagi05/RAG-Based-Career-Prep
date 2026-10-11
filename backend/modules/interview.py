import os


import pandas as pd
from openai import OpenAI
from llm import get_openai_client, model_name
import requests
import json
from dotenv import load_dotenv


load_dotenv()


def _openai_client() -> OpenAI:
    return get_openai_client()


def _google_nlp_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("NLP_KEY")




def setUp():
        selection = input("Do you want HR or Technical questions(Type HR or Technical): ")

        if selection == "HR":
                title = "Geothermal Production Managers"

                data = pd.read_csv("backend/modules/new_behavioral_interview_questions_dataset.csv")

                start_question = "Tell me about yourself"
                start_question_response = input(start_question + ": ")

                client = _openai_client()

                print(generate_feedback(start_question, start_question_response, client))

                next = input("Do you want another question? (Type yes or no): ")

                if next == "yes":
                        next_boolean = True

                        while next_boolean:
                                print("\n")

                                question = generate_questions(selection, data, title, client)

                                question_response = input(question + ": ")

                                print("\n")

                                print(generate_feedback(question, question_response, client))

                                print("\n")

                                next = input("Do you want another question? (Type yes or no): ")

                                if next == "no" or next != "yes":
                                        next_boolean = False

        if selection == "Technical":
                data = pd.read_csv("backend/modules/Task Statements.csv")

                title = "Geothermal Production Managers"

                start_question = "Tell me about yourself"
                start_question_response = input(start_question + ": ")

                client = _openai_client()

                print(generate_feedback(start_question, start_question_response, client))

                next = input("Do you want another question? (Type yes or no): ")

                if next == "yes":
                        next_boolean = True

                        while next_boolean:
                                print("\n")

                                question = generate_questions(selection, data, title, client)

                                question_response = input(question + ": ")

                                print("\n")

                                print(generate_feedback(question, question_response, client))

                                print("\n")

                                next = input("Do you want another question? (Type yes or no): ")

                                if next == "no" or next != "yes":
                                        next_boolean = False
   




def generate_questions(selection, data, title, client):
 
  list = []
  counter = 0


  if selection == "HR":
     prompt = f"From {data}, get a random interview question listed. Just the question. Don't say anything else."


     response = client.chat.completions.create(
      model=model_name("large"),
      messages=[
       {"role": "system", "content": "You are an expert interviewer."},
       {"role": "user", "content": prompt}
     ]
  )
     
     question = response.choices[0].message.content.strip()
     return question



  if selection == "Technical":
        prompt = f"Generate a one sentence {title} technical interview question based on these tasks: {data}. Just the question."


        response = client.chat.completions.create(
         model=model_name("large"),
         messages=[
       {"role": "system", "content": "You are an expert interviewer."},
       {"role": "user", "content": prompt}
     ]
  )
     
        question = response.choices[0].message.content.strip()
        return question




def generate_feedback(question, answer, client):
    prompt = f"Provide feedback on the following answer to the question '{question}': {answer}. Provide feedback on: 1) Relevance to the question, 2) Mention of key skills/experiences, 3) Clarity and structure. Suggest improvements."


    response = client.chat.completions.create(
        model=model_name("large"),
        messages=[
            {"role": "system", "content": "You are an expert interviewer."},
            {"role": "user", "content": prompt}
        ]
    )

    feedback_response = response.choices[0].message.content.strip()


    score = 0.0
    google_key = _google_nlp_key()
    if google_key:
        url = f"https://language.googleapis.com/v1/documents:analyzeSentiment?key={google_key}"

        payload = {
            "document": {
                "type": "PLAIN_TEXT",
                "content": answer
            },
            "encodingType": "UTF8"
        }

        try:
            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()
            sentiment_data = response.json()
            score = sentiment_data.get("documentSentiment", {}).get("score", 0.0)
        except (requests.RequestException, ValueError):
            score = 0.0


    # [LLM-JUDGE runtime] the large-tier LLM grades the *candidate's* answer on an anchored 0-1 scale.
    # It is blended below with Google NLP sentiment (60/40) so a confident-sounding but
    # empty answer can't score high on tone alone. The offline judge in evals/ grades
    # *this feedback* in turn (evals/run_evals.py --suite interview).
    quality_prompt = f"""Rate this interview answer quality from 0 to 1 where:
    1.0 = excellent, detailed, highly relevant
    0.5 = good but missing some points
    0.0 = poor, vague or irrelevant

    Answer: {answer}
    Question: {question}

    Return ONLY a number between 0 and 1, nothing else."""

    quality_response = client.chat.completions.create(
        model=model_name("large"),
        messages=[
            {"role": "system", "content": "You are an expert interview evaluator."},
            {"role": "user", "content": quality_prompt}
        ]
    )

    try:
        quality_score = float(quality_response.choices[0].message.content.strip())
        quality_score = max(0.0, min(1.0, quality_score))
    except:
        quality_score = 0.5

    
    combined_score = (quality_score * 0.6) + ((score + 1) / 2 * 0.4)

    if combined_score >= 0.6:
        sentiment = "positive"
    elif combined_score <= 0.4:
        sentiment = "negative"
    else:
        sentiment = "neutral"


    score = (combined_score * 2) - 1 



    onet_tasks = pd.read_csv(os.path.join(os.path.dirname(__file__), "Task Statements.csv"))


    prompt = f"From {answer}, find any key skills, experiences, or companies mentioned. Just list the skills or experiences, and nothing else. If you can't find anything just give a space."


    list = client.chat.completions.create(
        model=model_name("large"),
        messages=[
            {"role": "system", "content": "You are an expert interviewer."},
            {"role": "user", "content": prompt}
        ]
    )


    entities = list.choices[0].message.content.strip()


    prompt = f"From {answer}, find the strength that make this anser effective, in just one short sentence. Just that sentence."


    strength_response = client.chat.completions.create(
        model=model_name("large"),
        messages=[
            {"role": "system", "content": "You are an expert interviewer."},
            {"role": "user", "content": prompt}
        ]
    )

    strength = strength_response.choices[0].message.content.strip()


    prompt = f"From {answer}, find an improvement that can be made to make this answer more effective, in just one short sentence. Just that sentence."


    improvement_response = client.chat.completions.create(
        model=model_name("large"),
        messages=[
            {"role": "system", "content": "You are an expert interviewer."},
            {"role": "user", "content": prompt}
        ]
    )

    improvement = improvement_response.choices[0].message.content.strip()


    json_output = json.dumps({
        "question": question,
        "answer": answer,


        "feedback": {
            "sentiment": sentiment,
            "entities": entities,
            "strengths": strength,
            "improvements": improvement,
            "score": score
        }
    }, indent=2)



    return json_output


if __name__ == "__main__":
    setUp()

