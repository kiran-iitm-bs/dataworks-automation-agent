from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import os
import subprocess
import shutil
import urllib.request
import json
from datetime import datetime
from dateutil import parser
import httpx
import base64
import sqlite3
from prompt import INTERPRET_TASK_PROMPT
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from PIL import Image
import pytesseract

load_dotenv()

app = FastAPI()

class TaskRequest(BaseModel):
    task: str

API_KEY = os.getenv("AIPROXY_TOKEN")

def execute_query(query, params=()):
    """Executes a SQL query and returns the result."""
    db_path = "./data/ticket-sales.db"
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database not found")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query, params)
    result = cursor.fetchall()
    conn.close()
    return result

def api_call_to_llm(system: str, content: str, task="completions") -> str:
    """API calls to GPT-4o-mini"""

    # Make the API request
    if task == "completions":
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        endpoint = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        payload = {
            "model": "gpt-4o-mini",
            "messages":[
                {"role": "system", "content": system},
                {"role": "user", "content": content}
            ],
            "temperature": 0.0
        }
    elif task == "embeddings":
        endpoint = "http://aiproxy.sanand.workers.dev/openai/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {API_KEY}"
        }
        payload = {
            "model": "text-embedding-3-small",
            "input": content            
        }
    
    elif task == "vision":
        endpoint = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system},
                        {
                            "type": "image_url",
                            "image_url": {"url": content},
                        },
                    ],
                }
            ],            
        }

    response = httpx.post(url=endpoint, json=payload, headers=headers).json()

    if task == "completions":
        return response["choices"][0]["message"]["content"].strip()
    elif task == "embeddings":
        return response
    elif task == "vision":
        return response["choices"][0]

def interpret_task(task: str) -> str:
    """Interprets a task description and categorizes it."""
    return api_call_to_llm(system=INTERPRET_TASK_PROMPT, content=task, task="completions")

def execute_task(task: str) -> str:
    """Executes a given task and returns the result."""

    task_str = interpret_task(task)
    try:
        std_task = json.loads(task_str)
        if isinstance(std_task, dict) and "category" in std_task:
            pass
        else:
            return "Invalid JSON format from LLM."
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        raise Exception("Failed to parse JSON")

    try:
        if std_task["category"] == "install uv library and run datagen":
            user_email = std_task["argument"]
            if not shutil.which("uv"):
                subprocess.run(["pip", "install", "uv"], check=True)
            script_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
            subprocess.run(["uv", "run", script_url, user_email, "--root", "./data"], check=True)
            return "datagen.py executed successfully"
        
        elif std_task["category"] == "format using prettier":
            if not shutil.which("npx"):
                raise HTTPException(status_code=500, detail="npx not found")
            subprocess.run(["npx", "prettier@3.4.2", "--write", "./data/format.md"], check=True)
            return "File formatted successfully"
        
        elif std_task["category"] == "count wednesdays":
            date_file = "./data/dates.txt"
            output_file = "./data/dates-wednesdays.txt"
            if not os.path.exists(date_file):
                raise HTTPException(status_code=500, detail="File not found")
            
            with open(date_file, "r", encoding="utf-8") as file:
                dates = file.readlines()

            wednesday_count = 0

            for date_str in dates:
                parsed_date = parser.parse(date_str.strip())
                if parsed_date.weekday() == 2:
                    wednesday_count += 1
            
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(str(wednesday_count))
            
            return f"Counted {wednesday_count} Wednesdays and wrote to {output_file}"
        
        elif std_task["category"] == "sort array of contacts":
            contacts_file = "./data/contacts.json"
            sorted_file = "./data/contacts-sorted.json"
            if not os.path.exists(contacts_file):
                raise HTTPException(status_code=404, detail="File not found")
            
            with open(contacts_file, "r", encoding="utf-8") as file:
                contacts = json.load(file)
            
            sorted_contacts = sorted(contacts, key=lambda c: (c["last_name"], c["first_name"]))
            
            with open(sorted_file, "w", encoding="utf-8") as file:
                json.dump(sorted_contacts, file)
            
            return f"Sorted contacts and wrote to {sorted_file}"
        
        elif std_task["category"] == "write 10 most recent logs":
            logs_dir = "./data/logs"
            output_file = "./data/logs-recent.txt"
            
            if not os.path.exists(logs_dir):
                raise HTTPException(status_code=404, detail="Logs directory not found")
            
            log_files = sorted(
                (os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.endswith(".log")),
                key=os.path.getmtime,
                reverse=True
            )[:10]
            
            first_lines = []
            for log_file in log_files:
                with open(log_file, "r", encoding="utf-8") as file:
                    first_line = file.readline().strip()
                    first_lines.append(first_line)
            
            with open(output_file, "w", encoding="utf-8") as file:
                file.write("\n".join(first_lines))
            
            return f"Extracted first lines from recent logs and wrote to {output_file}"

        elif std_task["category"] == "find markdown files and extract h1 tags":
            docs_dir = "./data/docs"
            index_file = "./data/docs/index.json"
            
            if not os.path.exists(docs_dir):
                raise HTTPException(status_code=404, detail="Docs directory not found")
            
            index = {}
            for root, _, files in os.walk(docs_dir):
                for file in files:
                    if file.endswith(".md"):
                        file_path = os.path.join(root, file)
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.startswith("# "):
                                    title = line.strip()[2:]
                                    index[os.path.relpath(file_path, docs_dir)] = title
                                    break
            
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(index, f)
            
            return f"Created Markdown index at {index_file}"
        
        elif std_task["category"] == "extract sender email address":
            email_file = "./data/email.txt"
            output_file = "./data/email-sender.txt"
            
            if not os.path.exists(email_file):
                raise HTTPException(status_code=404, detail="File not found")
            
            with open(email_file, "r", encoding="utf-8") as file:
                email_content = file.read()
            
            system_message = "Extract the sender's email address from the given email content. Return just the email address."

            sender_email = api_call_to_llm(system_message, email_content)
            
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(sender_email)
            
            return f"Extracted sender's email and wrote to {output_file}"

        elif std_task["category"] == "extract credit card number":
            image_path = "./data/credit_card.png"
            output_file = "./data/credit-card.txt"
            
            if not os.path.exists(image_path):
                raise HTTPException(status_code=404, detail="Image not found")
            
            if not shutil.which("tesseract"):
                subprocess.run(["apt-get", "update"], check=True)
                subprocess.run(["apt-get", "install", "-y", "tesseract-ocr"], check=True)
            
            system_message = "Extract the credit card number from the given text. Return just the credit card number without any spaces."

            image = Image.open(image_path)
            extracted_text = pytesseract.image_to_string(image)

            credit_card_number = api_call_to_llm(system=system_message, content=extracted_text)
            
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(credit_card_number.strip())
            
            return f"Extracted credit card number and wrote to {output_file}"

        elif std_task["category"] == "find similar comments":
            comments_file = "./data/comments.txt"
            output_file = "./data/comments-similar.txt"
            
            if not os.path.exists(comments_file):
                raise HTTPException(status_code=404, detail="File not found")
            
            with open(comments_file, "r", encoding="utf-8") as file:
                comments = [line.strip() for line in file.readlines() if line.strip()]            
            
            response = api_call_to_llm(system="Find similar comments", content=comments, task="embeddings")["data"]

            embeddings = np.array([sample["embedding"] for sample in response])
            similarity = np.dot(embeddings, embeddings.T)
            # Create mask to ignore diagonal (self-similarity)
            np.fill_diagonal(similarity, -np.inf)
            # Get indices of maximum similarity
            i, j = np.unravel_index(similarity.argmax(), similarity.shape)
            result = "\n".join(sorted([comments[i], comments[j]]))
            
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(result)
            
            return f"Found most similar comments and wrote to {output_file}"

        elif std_task["category"] == "find total sales of 'gold' ticket type in the db table":
            query = "SELECT SUM(total_sales) FROM (SELECT (units * price) AS total_sales FROM tickets WHERE LOWER(TRIM(type))='gold')"
            result = execute_query(query)
            total_sales = result[0][0] if result[0][0] is not None else 0
            with open("./data/ticket-sales-gold.txt", "w") as f:
                f.write(str(total_sales))
            return {"message": "Total sales for Gold tickets computed successfully"}
        else:
            raise ValueError("Unknown task")
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run")
def run_task(task: str = Query(..., description="Task description")):
    """Executes a task and returns the output."""
    result = execute_task(task)
    return {"status": "success", "output": result}

@app.get("/read")
def read_file(path: str = Query(..., description="Path to the file")):
    """Returns the content of the specified file."""
    mod_path = "." + path
    if not os.path.exists(mod_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    with open(mod_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    return PlainTextResponse(content=f"""{content}""")
