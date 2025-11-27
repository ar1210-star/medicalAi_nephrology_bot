import logging
from typing import Any, Dict, Tuple, List
from app.llm.groq_client import call_groq_chat
from app.agents.receptionist import receptionist_agent
from app.agents.clinical import clinical_agent
from app.logging_setup import logger

State = Dict[str, Any]

ADMIN_KEYWORDS = [
    "appointment",
    "book an appointment",
    "book appointment",
    "schedule an appointment",
    "schedule appointment",
    "reschedule",
    "cancel appointment",
    "next appointment",
    "follow-up appointment",
    "follow up appointment",
    "slot",
    "time slot",
]

def _quick_admin_check(message: str) -> bool:
    text = message.lower()
    return any(k in text for k in ADMIN_KEYWORDS)



def _format_history_for_classifier(history: List[Dict[str, Any]]) -> str:
    """
    Take a short history list and format it into a compact text block
    for the classifier prompt.

    Each item in history is expected to look like:
      {"role": "user"/"assistant", "agent": "receptionist"/"clinical"/None, "content": str}
    """
    lines = []
    # only last 6 turns to keep context short
    for h in history[-6:]:
        role = h.get("role", "")
        agent = h.get("agent", "")
        content = h.get("content", "")
        if role == "user":
            lines.append(f"USER: {content}")
        else:
            if agent:
                lines.append(f"ASSISTANT({agent}): {content}")
            else:
                lines.append(f"ASSISTANT: {content}")
    return "\n".join(lines)


def _classify_intent(message: str, state: State) -> str:
    """
    Use a small LLM call to classify the user message.

    Labels:
      - IDENTITY  : user is telling or clarifying their name/who they are.
      - ADMIN     : appointments, scheduling, transport, documents, contact info.
      - CLINICAL  : symptoms, diagnosis, meds, side effects, diet for disease, labs, prognosis, etc.
      - SMALL_TALK: greetings, acknowledgements (yes/ok/thanks), generic chit-chat.

    Now uses short conversation history from state to understand context better.
    """
    patient_record = state.get("patient_record")
    history = state.get("history", [])

    history_text = _format_history_for_classifier(history)

    system_prompt = (
        "You are a router for a hospital chatbot.\n"
        "Your task is to classify the NEXT user message into exactly ONE label:\n"
        "- IDENTITY  : They are telling or clarifying their name or identity.\n"
        "- ADMIN     : They ask about appointments, scheduling, rescheduling, cancelling visits, "
        "transportation, documents, reports, contact numbers, or clinic timings.\n"
        "- CLINICAL  : They ask about symptoms, diagnosis, kidney disease, nephrotic syndrome, "
        "medications, side effects, diet for a disease, lab results, prognosis, or general medical information.\n"
        "- SMALL_TALK: Greetings, acknowledgements (yes/ok/good/thanks), or casual remarks "
        "without a concrete request.\n\n"
        "Important:\n"
        "- Use the conversation history to understand what 'yes', 'okay', or 'good' refers to.\n"
        "- If the current message is just a short acknowledgement (e.g. 'yes', 'okay', 'good', 'fine') "
        "after a greeting or explanation, treat it as SMALL_TALK.\n"
        "- If they ask to book, reschedule, or confirm an appointment, choose ADMIN.\n"
        "- If there is ANY substantial medical/clinical content, choose CLINICAL.\n"
        "- Reply with ONLY the label: IDENTITY, ADMIN, CLINICAL, or SMALL_TALK."
    )

    context_line = ""
    if patient_record:
        context_line = (
            f"Known patient diagnosis: {patient_record.get('primary_diagnosis')}, "
            f"discharge date: {patient_record.get('discharge_date')}"
        )

    user_prompt = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Context (if any): {context_line}\n\n"
        f"Next user message: {message}\n\n"
        "Answer with exactly one word: IDENTITY, ADMIN, CLINICAL, or SMALL_TALK."
    )

    label = call_groq_chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model="openai/gpt-oss-20b",
        temperature=0.0,
        max_tokens=5,
    ).strip().upper()

    if label not in {"IDENTITY", "ADMIN", "CLINICAL", "SMALL_TALK"}:
        # Fallback: if we know the patient already, default to CLINICAL; otherwise IDENTITY
        if state.get("patient_record"):
            return "CLINICAL"
        return "IDENTITY"

    return label


def handle_message(message: str, state: State) -> Tuple[str, State]:
    allow_web = state.get("allow_web", True)
    patient_record = state.get("patient_record")

    history: List[Dict[str, Any]] = state.get("history", [])
    state["history"] = history
    history.append({"role": "user", "agent": None, "content": message})
    session_id = state.get("session_id", "unknown") 

    # 1) No identity yet → receptionist
    if not patient_record:
        logger.info(
            "ROUTER session_id=%s route=receptionist reason=no_identity message=%s",
            session_id,
            message[:200],
        )
        reply, state, _ = receptionist_agent(message, state)
        state["mode"] = "receptionist"
        state["history"].append(
            {"role": "assistant", "agent": "receptionist", "content": reply}
        )
        return reply, state

    # 2) HARD OVERRIDE: if clearly appointment-related, treat as ADMIN
    if _quick_admin_check(message):
         logger.info(
            "ROUTER session_id=%s route=receptionist reason=quick_admin_match message=%s",
            session_id,
            message[:200],
        )
         reply, state, _ = receptionist_agent(message, state)
         state["mode"] = "receptionist"
         state["history"].append(
            {"role": "assistant", "agent": "receptionist", "content": reply}
        )
         return reply, state

    # 3) Otherwise, use LLM classifier with history
    intent = _classify_intent(message, state)
    logger.info(
        "ROUTER session_id=%s intent=%s allow_web=%s message=%s",
        session_id,
        intent,
        allow_web,
        message[:200],
    )

    if intent in {"ADMIN", "IDENTITY"}:
        reply, state, _ = receptionist_agent(message, state)
        state["mode"] = "receptionist"
        state["history"].append(
            {"role": "assistant", "agent": "receptionist", "content": reply}
        )
        logger.info(
            "ROUTER session_id=%s final_agent=receptionist intent=%s", session_id, intent
        )
        return reply, state

    # 4) SMALL_TALK + CLINICAL → clinical
    reply, state = clinical_agent(message, state, allow_web=allow_web)
    state["mode"] = "clinical"
    state["history"].append(
        {"role": "assistant", "agent": "clinical", "content": reply}
    )
    logger.info(
        "ROUTER session_id=%s final_agent=clinical intent=%s", session_id, intent
    )
    return reply, state
