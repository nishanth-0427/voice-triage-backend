# Voice-Triage: AI Sports Safety for Grassroots Coaches

## Problem Statement
Youth sports programs in underserved communities lack access to elite sports technology. This API allows grassroots coaches to submit basic voice notes of player exertion or physical complaints, returning instant, AI-driven injury risk flags and triage protocols.

## Tech Stack
* **Backend:** FastAPI (Python)
* **AI Core:** Google Gemini 3.5 Flash (Audio processing & LLM triage)
* **Integration:** Webhook architecture

## Setup Instructions (Local)
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Set your environment variable: `$env:GEMINI_API_KEY="your_api_key"`
4. Run the server: `uvicorn main:app --reload`
5. Test the webhook using the provided `test_webhook.py` script.
