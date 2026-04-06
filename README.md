# ClearPath Voicebot

Clinical triage voice assistant backend and dashboard.

# NOTE!!!! to test this system, placing a phone call is the best way experience it, due to credit preservation you will have to get in touch with me so I can provide that number. Also there is a paper that you can access to read more about this and also a youtube video(https://youtu.be/zAsvl8DbEgU). Thank you for exploring this.

## Project Overview
This project receives call webhooks, classifies urgency (Category 1/2/3), stores call records, and displays them in a Streamlit dashboard.

Core flow:
- Voice call webhook -> FastAPI backend
- Triage classification and action recommendation
- Call log persistence in SQLite
- Streamlit dashboard for monitoring call history

## Tech Stack
- Python 3.12+
- FastAPI
- Streamlit
- SQLite
- Pandas

## Folder Structure
- `backend/app.py`: FastAPI backend and webhook logic
- `backend/dashboard.py`: Streamlit dashboard
- `backend/triage_rules.json`: symptom triage rules
- `backend/requirements.txt`: Python dependencies
- `docs/ClearPath_Health_paper.pdf`: project paper/report

## Project Paper
The project paper is included in this repository at:
- `docs/ClearPath_Health_paper.pdf`

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r backend/requirements.txt`

## Run Backend
From project root:
- `cd backend`
- `python app.py`

Backend URL:
- `http://127.0.0.1:8000`

## Run Dashboard
From project root:
- `cd backend`
- `python -m streamlit run dashboard.py --server.port 8501`

Dashboard URL:
- `http://127.0.0.1:8501`

## Notes for Evaluation
- The webhook endpoint is available at `/webhook`.
- Duplicate end-of-call events are de-duplicated using `call_id` in SQLite.
- Transcript capture excludes system prompt lines from stored call transcripts.

## Submission Contents
This submission excludes local runtime artifacts and secrets (such as `.env`, local databases, debug logs, and `venv`).
