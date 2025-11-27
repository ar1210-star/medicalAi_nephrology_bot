# Nephrology Post-Discharge AI Assistant

This project implements a conversational medical-support system for kidney-related patients after hospital discharge. It simulates a real hospital digital assistant with two specialized agents: a friendly receptionist agent for administrative support and a clinical agent for medical information. It also incorporates medical RAG (retrieval-augmented generation) using nephrology domain text, and optional web-search for up-to-date research references.

---

## What the System Does

When a user interacts with the assistant:

* It verifies the user by name using discharge-record data.
* It answers administrative questions as a receptionist.
* When the user asks something clinical, it transfers the conversation to the clinical agent.
* The clinical agent answers medically relevant questions using a domain-specific knowledge base (vector DB + embeddings).
* If the information is not available locally, the system can fall back to web search.
* All interactions flow through a FastAPI backend.
* A Streamlit interface provides a chat-like front-end experience.

---

## Key Components

### Receptionist Agent (administrative / non-medical)

Handles:

* identity confirmation
* appointment scheduling support
* transportation and paperwork assistance
* discharge date confirmation
* medication pickup logistics
  If the user asks a clinical question, the receptionist politely declines and forwards it to the clinical agent.

### Clinical Agent (medical)

Handles:

* explanations of symptoms
* causes of kidney dysfunction
* chronic kidney disease information
* nephrotic syndrome information
* medication explanations
* pathology of CKD progression
* edema / swelling causes
  Uses:
* local textbook-based RAG
* optional web search for newer research
* medical disclaimers when necessary

---

## Knowledge Base (RAG Storage)

The system uses vector embeddings created from nephrology textbook content (`Comprehensive Clinical Nephrology`).
Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
Vector store: persistent ChromaDB

This allows the model to retrieve medically authoritative knowledge rather than hallucinate.

---

## Backend Architecture

* FastAPI handles message routing.
* Orchestrator determines which agent should respond.
* Contains a message classifier to categorize user messages as:

  * IDENTITY
  * ADMINISTRATIVE
  * CLINICAL
  * SMALL_TALK
* Each message, agent selection, and final response is logged.

---

## Frontend

* Implemented using Streamlit
* Provides a ChatGPT-like UI
* Includes a toggle for enabling/disabling web search
* Connects to the FastAPI backend message endpoint

---

## Running Locally

Clone and enter project:

```
git clone https://github.com/ar1210-star/medicalAi_nephrology_bot.git
cd medicalAi_nephrology_bot
```

Create virtual environment:

```
python -m venv venv
```

Activate:

```
venv\Scripts\activate   (Windows)
```

Install dependencies:

```
pip install -r requirements.txt
```

Create `.env` file containing:

```
GROQ_API_KEY=xxxxxx
TAVILY_API_KEY=xxxxxx  (optional)
```

Run backend:

```
uvicorn app.api:app --reload
```

Run frontend:

```
streamlit run app/ui/chat_app.py
```

---

## Example Interaction

**User**: hi I am John Smith
**Receptionist**: Welcome back John Smith…

**User**: what causes swelling in legs?
**Receptionist**: This seems like a medical concern; connecting you to our clinical agent.
**Clinical agent**: Leg swelling (edema) in CKD can occur due to…

---

## Logging

All chat flow is logged in:

```
logs/app.log
```

Logged information includes:

* session ID
* intent classification
* chosen agent
* message content
* routing decisions

---

## Disclaimer

This system is strictly for educational demonstration.
It does not provide medical diagnoses or treatment instructions and does not replace a real physician.

---

## Future Extensions

* real patient authentication
* integration with hospital EMR systems
* physician override mode
* multilingual interface
* voice-based interaction

