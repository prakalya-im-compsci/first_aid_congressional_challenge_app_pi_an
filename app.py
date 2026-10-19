# Import necessary modules from Flask and Google GenAI
import os
from flask import Flask, render_template, request, jsonify
from google import genai
from dotenv import load_dotenv

# Load environment variables from the .env file (keeps API key secure)
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)

# Fetch the API key securely from the system environment
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the official Google GenAI client with the secure key
client = genai.Client(api_key=api_key)

# Define the system instructions for the AI's clinical behavior
SYSTEM_INSTRUCTION = """
You are a strict, calm, clinical emergency first-aid assistant managing an interactive, step-by-step response.
Provide ONLY ONE action step at a time. 
If the user says they cannot do a step or lack a supply, immediately adapt with a safe alternative.
Format your response separated by the marker "---ASCII---":
Part 1: Clean ASCII art icon illustrating the current action (max 5 lines).
Part 2: The direct instruction or alternative.
"""

@app.route("/")
def home():
    """Serves the main HTML frontend page when users visit the app."""
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    """Endpoint that receives user input from the frontend, talks to Gemini securely, and returns the response."""
    data = request.json
    user_message = data.get("message", "")
    
    try:
        # Call the Gemini 2.5 Flash model using the official SDK
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_message,
            config={
                'system_instruction': SYSTEM_INSTRUCTION,
                'temperature': 0.2, # Low temperature for consistent, factual safety output
            }
        )
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"response": "---ASCII---\n[ERROR]\n---ASCII---\nConnection issue. Ensure 911 is called if life-threatening."}), 500

# Run the local server
if __name__ == "__main__":
    app.run(debug=True, port=5000)