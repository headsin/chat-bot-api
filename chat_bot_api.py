from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import pdfplumber
import numpy as np
import requests
from bs4 import BeautifulSoup
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Azure OpenAI Setup
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint="https://job-recruiting-bot.openai.azure.com/"
)
chat_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# In-memory session storage (for production, use Redis or database)
sessions = {}

# Utility Functions
def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def clean_text(text):
    return text.replace('\n', ' ').replace('\r', ' ').strip()

def chunk_text(text, chunk_size=700):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def scrape_page(url):
    try:
        res = requests.get(url, timeout=60)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        return clean_text(soup.get_text(separator=' ', strip=True))
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""

# Prepare embeddings (cached globally)
def prepare_embeddings():
    urls = [
        "https://headsin.co/",
        "https://headsin.co/code-of-conduct",
        "https://headsin.co/privacy-policy",
        "https://headsin.co/terms-and-conditions",
        "https://headsin.co/about-us",
        "https://headsin.co/contact-us",
        "https://headsin.co/build-resume-page",
        "https://headsin.co/candidate",
        "https://headsin.co/company"
    ]

    web_text = "\n".join([scrape_page(url) for url in urls])

    try:
        with pdfplumber.open("./HeadsIn_Public_Chatbot_Report_2025.pdf") as pdf:
            pdf_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    except FileNotFoundError:
        print("PDF file not found, continuing without PDF content")
        pdf_text = ""

    combined = web_text + "\n" + pdf_text
    chunks = [clean_text(c) for c in chunk_text(combined, chunk_size=700) if c.strip()]

    embeddings = []
    for idx, chunk in enumerate(chunks):
        resp = client.embeddings.create(model=embedding_deployment, input=[chunk])
        embeddings.append({"text": chunk, "embedding": resp.data[0].embedding})
    return embeddings

# Load embeddings on startup
print("Loading embeddings...")
chunk_embeddings = prepare_embeddings()
print(f"Loaded {len(chunk_embeddings)} embeddings")

FINAL_BLOCK_MESSAGE = (
    "Thanks for checking with us. Feel free to visit our website.\n\n"
    "If you're seeking a job, visit: <a href=\"https://headsin.co/auth\" target=\"_blank\">https://headsin.co/auth</a> \n\n"
    "If you're looking to hire candidates, go to: <a href=\"https://company.headsin.co/auth\" target=\"_blank\">https://company.headsin.co/auth</a> \n\n"
    "For Further Details, Contact: \n\n"
    "Call : <a href=\"tel:+919773497763\">+91 97734 97763</a> \n\n"
    "Email : <a href=\"mailto:info@headsin.co\">info@headsin.co</a> \n"
)


def get_or_create_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [("assistant", "Hello! How can I help you today?")],
            "irrelevant_count": 0,
            "created_at": datetime.now()
        }
    return sessions[session_id]

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "HeadsIn API is running", "timestamp": datetime.now().isoformat()})


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/chat', methods=['POST'])
def chat():
    api_key_header = request.headers.get("X-API-KEY")
    if api_key_header != os.getenv("CHAT_API_KEY"):
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({"error": "Message is required"}), 400
        
        user_input = data['message']
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        # Get or create session
        session = get_or_create_session(session_id)
        
        # Check if session is blocked due to irrelevant questions
        if session['irrelevant_count'] >= 2:
            return jsonify({
                "response": FINAL_BLOCK_MESSAGE,
                "session_id": session_id,
                "blocked": True,
                "history": session['history']
            })
        
        cleaned_input = clean_text(user_input)
        
        # Get embedding for user input
        user_emb = client.embeddings.create(model=embedding_deployment, input=[cleaned_input])
        user_vec = user_emb.data[0].embedding
        
        # Find best matching chunk
        best_chunk = max(chunk_embeddings, key=lambda c: cosine_similarity(user_vec, c["embedding"]))["text"]
        
        # System prompt
        system_prompt = (
            "IMPORTANT FORMATTING RULES (ALWAYS FOLLOW):\n"
            "- ALL links must be HTML clickable links.\n"
            "  Example: https://headsin.co/auth → <a href=\"https://headsin.co/auth\" target=\"_blank\">https://headsin.co/auth</a>\n"
            "- ALL emails must be HTML mailto links.\n"
            "  Example: info@headsin.co → <a href=\"mailto:info@headsin.co\">info@headsin.co</a>\n"
            "- ALL phone numbers must be HTML tel links with correct format.\n"
            "  Example: +91 97734 97763 → <a href=\"tel:+919773497763\">+91 97734 97763</a>\n"
            "- Do NOT output Markdown links. Only use HTML <a> tags.\n\n"

            "You are a professional, friendly, and concise AI assistant for HeadsIn, an AI-powered job search and hiring platform based in India.\n\n"

            "Guidelines:\n"
            "- ALWAYS reply in 1–2 short sentences. Do not write paragraphs.\n"
            "- You help users with job search, hiring, resumes, assessments, interview process, and HeadsIn related questions.\n"
            "- Greet users warmly when they say 'hi', 'hello', or introduce themselves (e.g., 'I am Renish').\n"
            "- If a user is looking for a job (e.g., role, location), guide them on how to apply via HeadsIn.\n"
            "- If a recruiter wants to hire, guide them to post a job using the platform.\n"
            "- If a question seems unrelated, respond once politely, and stop after 2 unrelated messages.\n"

            "Special Instructions:\n"
            "- If asked: 'What is HeadsIn?'\n"
            "  → HeadsIn is an AI-powered job platform that matches candidates to jobs and helps recruiters hire faster.\n"
            "- If asked: 'How do I apply for a job?' or any related question:\n"
            "  → Go to https://headsin.co/auth, log in, create your resume, complete a short assessment, and apply.\n"
            "- If asked: 'How do I post a job?' or how to find candidates or any related question:\n"
            "  → Visit https://company.headsin.co/auth, log in as a recruiter, fill job details, and post.\n"
            "- If asked: 'In how many days will I get a job?'\n"
            "  → It depends on your profile and company requirements.\n"
            "- If asked about support:\n"
            "  → Contact us at info@headsin.co, call 9773497763, or use https://headsin.co/contact-us.\n"
            "- If asked about social media:\n"
            "  → Instagram: https://www.instagram.com/headsin.co | Facebook: https://www.facebook.com/people/HeadsInco/61574907748702/ | LinkedIn: https://www.linkedin.com/company/headsinco/\n"

            "Irrelevant Question Rule:\n"
            "- If the user's message is clearly unrelated to job search, hiring, or HeadsIn, politely say: 'I'm sorry, I can only assist with questions related to the HeadsIn platform.'\n"
            "- If this happens 3 times, reply once more and stop responding:\n"
            "  'Thanks for checking with us. For more information, visit:\n"
            "  Job Seekers: https://headsin.co/auth\n"
            "  Recruiters: https://company.headsin.co/auth'\n"
            "Any Other Questions:\n"
            "   Call: +91 9773497763\n"
            "   Email: info@headsin.co\n"
        )

        
        # Send to AI
        chat_response = client.chat.completions.create(
            model=chat_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Use the following to answer the user query:\n\nRelevant info: {best_chunk}\n\nUser: {user_input}"}
            ]
        )
        answer = chat_response.choices[0].message.content.strip()
        
        # Handle irrelevant detection
        irrelevant_phrases = [
            "i'm sorry", "i can only assist", "only assist with", "not related to headsin"
        ]
        if any(phrase in answer.lower() for phrase in irrelevant_phrases):
            session['irrelevant_count'] += 1
        else:
            session['irrelevant_count'] = 0
        
        if session['irrelevant_count'] >= 3:
            answer = FINAL_BLOCK_MESSAGE
        
        # Store message history
        session['history'].append(("user", user_input))
        session['history'].append(("assistant", answer))
        
        return jsonify({
            "response": answer,
            "session_id": session_id,
            "blocked": session['irrelevant_count'] >= 2,
            "history": session['history']
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/session/<session_id>', methods=['GET'])
def get_session(session_id):
    if session_id in sessions:
        return jsonify({
            "session_id": session_id,
            "history": sessions[session_id]['history'],
            "irrelevant_count": sessions[session_id]['irrelevant_count'],
            "blocked": sessions[session_id]['irrelevant_count'] >= 2
        })
    else:
        return jsonify({"error": "Session not found"}), 404

@app.route('/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    if session_id in sessions:
        del sessions[session_id]
        return jsonify({"message": "Session deleted successfully"})
    else:
        return jsonify({"error": "Session not found"}), 404

@app.route('/sessions', methods=['GET'])
def list_sessions():
    return jsonify({
        "sessions": list(sessions.keys()),
        "total": len(sessions)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
