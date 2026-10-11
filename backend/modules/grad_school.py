import os
import asyncio
import chromadb
from typing import List, Dict
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from database import mongo
from dotenv import load_dotenv
from llm import get_openai_client, model_name

# Load env 
load_dotenv()

# ChromaDB set up
chroma_client = chromadb.PersistentClient(path="./chroma_db")

try:
    grad_collection = chroma_client.get_collection(name="grad_programs")
except:
    grad_collection = chroma_client.create_collection(name="grad_programs")



# Grad School Prep Module
class GradSchoolPrep:
    def __init__(self):
        self.collection = grad_collection
        self.openai_client = get_openai_client()

    async def load_data_if_empty(self):
        if self.collection.count() == 0:
            print("Loading data into ChromaDB...")
            cursor = mongo.db["grad_requirement"].find()
            async for doc in cursor:
                doc_id = str(doc.get("_id"))

                text = f"""
                University: {doc.get("university", "")}
                Program: {doc.get("program", "")}
                GPA Requirement: {doc.get("gpa", "")}
                GRE Requirement: {doc.get("gre", "")}
                Prerequisites: {doc.get("prerequisites", "")}
                Deadline: {doc.get("deadline", "")}
                Description: {doc.get("description", "")}
                """

                # Clean metadata 
                metadata = {
                    "university": str(doc.get("university", "")),
                    "program": str(doc.get("program", "")),
                    "gpa": str(doc.get("gpa", "")),
                    "gre": str(doc.get("gre", "")),
                    "deadline": str(doc.get("deadline", "")),
                    "description": str(doc.get("description", ""))
                }

                self.collection.add(
                    ids=[doc_id],
                    documents=[text],
                    metadatas=[metadata]
                )
            print(f"Added {self.collection.count()} programs to ChromaDB")
        else:
            print(f"ChromaDB already loaded ({self.collection.count()} programs)")
    
    # Fit score
    def compute_fit(self, student, program):
        score = 0
        total = 3

        # GPA
        min_gpa = float(program.get("gpa", 0) or 0)
        if student["gpa"] >= min_gpa:
            score += 1

        # GRE (optional)
        gre_req = program.get("gre", 0)
        if gre_req:
            if student.get("gre", 0) >= float(gre_req):
                score += 1
        else:
            score += 1  # no GRE required = full credit

        # Coursework match
        prereqs = program.get("prerequisites", [])
        if isinstance(prereqs, list) and len(prereqs) > 0:
            match = sum(1 for c in prereqs if c in student["coursework"])
            score += match / len(prereqs)
        else:
            score += 1

        return round(score / total, 2)

    # SOP Generator 
    def generate_sop(self, student, program):
        prompt = f"""
        Student Profile:
        Major: {student['major']}
        GPA: {student['gpa']}
        Coursework: {", ".join(student['coursework'])}

        Program:
        {program.get("program")} at {program.get("university")}
        Description: {program.get("description")}

        Write a concise Statement of Purpose paragraph explaining why the student is a strong fit.
        Mention specific coursework and align it with the program focus.
        """

        response = self.openai_client.chat.completions.create(
        model=model_name("small"),
        messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content.strip()

    # Main pipeline
    async def run(self, student_profile):
        await self.load_data_if_empty()

        print("\nSearching programs...")

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.collection.query(
                query_texts=[student_profile["major"]],
                n_results=10
            )
        )

        programs = []

        for i in range(len(results["documents"][0])):
            metadata = results["metadatas"][0][i]

            fit = self.compute_fit(student_profile, metadata)
            sop = self.generate_sop(student_profile, metadata)

            programs.append({
                "university": metadata.get("university", "Unknown"),
                "program": metadata.get("program", "Unknown"),
                "fit_score": fit,
                "requirements": {
                    "gpa": metadata.get("gpa", "N/A"),
                    "gre": metadata.get("gre", "N/A"),
                    "deadline": metadata.get("deadline", "N/A")
                },
                "sop": sop
            })

        # Remove duplicates based on university + program
        unique_programs = {}

        for p in programs:
            key = (p["university"], p["program"])
            if key not in unique_programs or p["fit_score"] > unique_programs[key]["fit_score"]:
                unique_programs[key] = p

        programs = list(unique_programs.values())

        # Sort and take top 5
        programs = sorted(programs, key=lambda x: x["fit_score"], reverse=True)[:5]

        return programs


# Test run
if __name__ == "__main__":
    async def test():
        module = GradSchoolPrep()
        student = {
            "major": "Computer Science",
            "gpa": 3.5,
            "gre": 310,
            "coursework": ["Machine Learning", "Data Structures", "Databases"]
        }
        results = await module.run(student)  # ← await it
        for i, p in enumerate(results, 1):
            print(f"{i}. {p['university']} - {p['program']}")
            print(f"   Fit Score: {p['fit_score']}")
            print(f"   GPA Req: {p['requirements']['gpa']} | GRE Req: {p['requirements']['gre']} | Deadline: {p['requirements']['deadline']}")
            print(f"   SOP:\n   {p['sop']}")
            print()

    asyncio.run(test())