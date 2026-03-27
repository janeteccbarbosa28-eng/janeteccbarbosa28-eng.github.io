# Flask server that receives messages from the portfolio website
# and calls the Groq API securely (API key never exposed to the browser)
# Uses LLaMA 3.3 70B via Groq

from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Allow requests from GitHub Pages and localhost for development
CORS(app, origins=[
    "https://janeteccbarbosa28-eng.github.io",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:3000"
])

# Load Groq client using API key from environment variable
# NEVER hardcode the API key here — use environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API")

client = Groq(api_key=GROQ_API_KEY)

# System prompt with Janete's full context
SYSTEM_PROMPT = """You are Janete's AI Assistant on her personal portfolio website.
Answer questions about Janete Barbosa concisely and professionally. Keep answers under 120 words.

KEY FACTS:
- Junior Data Scientist & ML Engineer based in Portugal
- Background: Bachelor's Degree in Management (University of Algarve, 2019-2022) + nearly 3 years at Viking River Cruises (Guest Services + IT Support)
- Education: IronHack Data Science & ML Bootcamp (Jan-Mar 2026, 400+ hours, 6 projects)
- Looking for: Junior Data Scientist, ML Engineer, AI Engineer, or Data Analyst roles
- Open to: Remote, hybrid, on-site across Europe and Africa (personal connection to Angola)
- Languages: Portuguese (C2), English (B2), Spanish (B2)
- Contact: janete.cc.barbosa.28@gmail.com | linkedin.com/in/janete-barbosa-datascientist

PROJECTS:
1. Guest Intelligence System (Final Project) — Hybrid Recommender System (Content-Based + WBPR Collaborative Filtering) with LLM-powered Streamlit dashboard. NDCG@10=0.84. Tech: Python, LangChain, RAG, Cornac, Streamlit, SQL
2. Fake & Real News Classifier — NLP pipeline with fine-tuned DistilBERT achieving F1=0.98. LLaMA 3.3 chatbot via Groq API. Tech: Python, HuggingFace, DistilBERT, Streamlit
3. CIFAR-10 Image Classification — 96.94% accuracy with stacked EfficientNet models. Tech: Python, TensorFlow, PyTorch, Keras
4. House Price Prediction — Linear and logarithmic regression models. Tech: Python, Scikit-learn, Pandas
5. Airbnb EDA & SQL — Analysis of 102,599 listings. Tech: Python, SQL, Pandas, Seaborn
6. Vikings OOP Game — Turn-based combat simulation. Tech: Python OOP

TECH STACK: Python, Pandas, NumPy, Scikit-learn, PyTorch, TensorFlow, LangChain, RAG, LLMs, HuggingFace, SQL, Power BI, Tableau, Streamlit, DistilBERT, EfficientNet, Groq API

If asked about CV/resume, direct them to the Request CV button in the Resume section.
If asked about hiring or opportunities, encourage them to use the Hire Me button or email directly.
Always be friendly, professional and concise."""


@app.route("/chat", methods=["POST"])
def chat():
    """Receive a message from the website and return a response from LLaMA 3.3."""

    # Get the message from the request body
    data = request.get_json()

    # Validate that message exists
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"].strip()

    # Reject empty messages
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    try:
        # Call the Groq API with LLaMA 3.3
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=300,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )

        # Extract the reply text
        reply = response.choices[0].message.content

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": "Something went wrong. Please try again."}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint — used by Render to verify the server is running."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Run locally on port 5000 for development
    app.run(debug=True, port=5000)
