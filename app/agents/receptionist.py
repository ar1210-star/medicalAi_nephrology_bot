import os
# from langchain_groq import ChatGroq
# from langchain_core.messages import HumanMessage, SystemMessage
from app.tools.patient_db import find_patient_by_name 
from typing import List,Dict,Optional,Any,Tuple
from app.llm.groq_client import call_groq_chat
from dotenv import load_dotenv
load_dotenv()
State = Dict[str, Any]


# # receptionist_llm = ChatGroq(
#   model="openai/gpt-oss-20b",
#   temperature=0.4,
#   groq_api_key=os.getenv("GROQ_API_KEY")
# )

MEDICAL_KEYWORDS = [ #Initializing some medical keywords
   "pain", "swelling", "shortness of breath", "breathless",
    "bp", "blood pressure", "fever", "vomiting", "nausea",
    "chest pain", "palpitations", "urine", "urination",
    "weight gain", "weight loss", "dizzy", "dizziness",
    "headache", "cramps", "edema", "dialysis",
    "emergency", "urgent", "bleeding",
    "kidney", "renal", "ckd", "creatinine", "gfr",
    "stone", "stones",
    "low potassium", "low pottasium", "potassium", "pottasium",
    "low sodium", "low salt", "fluid restriction", "phosphorus", "phosphate",
	"protein restriction"
]

MEDICAL_TOPIC_WORDS = [ #Initializing some medical topic words
    "symptom", "symptoms", "symptomps", 
    "cause", "causes",
    "treatment", "treatments",
    "risk", "risks",
    "complication", "complications",
    "diet", "uses", "use", "effect", "effects",
    "side effect", "side effects",
    "benefit", "benefits",
    "management", "manage", "control",
]

def _extract_name(message: str) -> Optional[str]:
    """ Extract patient name from a simple intro sentence.
    """
    
    text = message.strip()
    lower = text.lower()
    prefixes = ["my name is", "i am", "this is", "name is", "i'm", "it's", "it is","hi i am","hello i am"]#prefixes for finding name
    for m in prefixes:
        idx = lower.find(m)
        if idx != -1:
            candidate = text[idx + len(m):].strip(" .,:;!-").strip() #retriving name
            if candidate:
                return candidate

    if " " not in text and len(text) > 1:
        return text

    return None
  

def _is_medical_query(message: str) -> bool:
    """
    A function for checking wether it is medical_query or not by checking medical keywords and topics
    """
    text = message.strip().lower()


    if "medical" in text or "symptom" in text or "symptomps" in text:
        return True

    for keyword in MEDICAL_KEYWORDS:
        if keyword in text:
            return True

    has_kidney_concept = any(
        kw in text
        for kw in ["kidney", "renal", "ckd", "stone", "stones", "creatinine", "gfr"]
    )
    has_topic = any(tw in text for tw in MEDICAL_TOPIC_WORDS)

    if has_kidney_concept and has_topic:
        return True


    return False

  
def receptionist_agent(message: str, state: State) -> Tuple[str, State, bool]:
	"""
	Handle human messages at the receptionist level.
	
	"""
    
	patient_name = state.get("patient_name") #retriving patient details from state
	patient_record = state.get("patient_record")
	conv_stage = state.get("conv_stage", 0)
	greeted = state.get("greeted", False)
    
	if state.get("awaiting_patient_disambiguation"): 
		#checking if there are two similar names and asking for discharge date if it contains ambiguity
		date_text = message.strip()
		candidates: List[Dict[str,Any]] = state.get("candidate_patients", []) #initializing candidates which should contain list of dictionaries
		match: Optional[Dict[str,Any]] = None
		for p in candidates:
			if p.get("discharge_date") == date_text:#checking for matching discharge date 
				match = p
				break
		if match:
			state["patient_name"] = match.get("patient_name")
			state["patient_record"] = match
			state["awaiting_patient_disambiguation"] = False #changing the value of ambiguity to false
			state["candidate_patients"] = []
			
			reply = ( #reply for matching patient
				"Thank you, I’ve confirmed your identity.\n\n"
                f"Welcome back {match.get('patient_name')}! You were discharged on "
                f"{match.get('discharge_date')} with the diagnosis "
                f"{match.get('primary_diagnosis')}.\n\n"
                "How are you feeling today?"
			)
			return reply, state, False

		reply = ( #reply if date also doesnt matching
			"I couldn’t match the discharge date you mentioned with our records. "
            "Please double-check the exact date on your discharge summary "
            "(for example: 2025-02-03), or share another detail."
		)
		return reply, state, False
		
	if not patient_name: #patient_name is empty
		extracted = _extract_name(message)#calling extract_name function for extracting the name of patient from message which contains prefixes
  
		if extracted:
			matches = find_patient_by_name(extracted)#finding patient records by name

			if not matches:
				reply = ( #replying if there is no name
					f"I couldn’t find any patient named '{extracted}'. "
                    "Please check the spelling and tell me your full name "
                    "as shown on the discharge summary."
				)
				return reply, state, False

			if len(matches)>1: #if there are more than 1 matches 
				state["awaiting_patient_disambiguation"] = True #convert ambiguity_state to true
				state["candidate_patients"] = matches
				reply = ( #replying if there is an ambiguity
					 f"There are multiple patients named '{extracted}' in our records. "
                    "Please tell me your discharge date so I can confirm your identity."
				)
				return reply, state, False

			record = matches[0] # retrieving patient details if there is only one record
			state["patient_name"] = record.get("patient_name")
			state["patient_record"] = record
   
			reply = ( # replying with patient details
				f"{record.get('patient_name')}! I see you were discharged on "
                f"{record.get('discharge_date')} with a primary diagnosis of "
                f"{record.get('primary_diagnosis')}.\n\n"
			)
			return reply, state, False

		reply = ( #Reply if the patient doesnt give any name
			"Welcome to our medical facility! To assist you better, "
			"please tell me your full name as shown on your discharge summary."
		)
		return reply, state, False

	if _is_medical_query(message): #checking if the query is medical or not
		# Forward to medical agent
		reply = (
			"Thanks for telling me that. Since this sounds like a medical concern, "
            "I’ll pass your message to our Clinical AI assistant. It will use your "
            "discharge details and medical reference material to give a more detailed answer.\n\n"
            "One moment while I hand this over."
		)
		return reply, state, True

	# Non-medical query – respond using Groq LLM
	summary = ( #creating summary by converting patient details into a single string
            f"Patient name: {patient_record.get('patient_name')}\n"
            f"Diagnosis: {patient_record.get('primary_diagnosis')}\n"
            f"Discharge date: {patient_record.get('discharge_date')}\n"
            f"Medications: {patient_record.get('medications')}\n"
            f"Dietary restrictions: {patient_record.get('dietary_restrictions')}\n"
            f"Follow-up: {patient_record.get('follow_up')}\n"
    )
	system_prompt = ( # describing system prompt
         "You are a hospital receptionist for recently discharged patients.\n"
        "- You have access to basic discharge details (diagnosis, discharge date, "
        "  medications, diet advice, follow-up plan).\n"
        "- You handle ONLY non-medical tasks: appointments, transport, documents, "
        "  contact details, simple check-in questions.\n"
        "- You MUST NOT give medical advice, interpret symptoms, or explain diseases.\n"
        "- If the message sounds clinical (symptoms, diagnosis, causes, treatment, diet "
        "  for a disease), respond briefly that the clinical assistant will handle the "
        "  medical details, and then ask if they need any non-medical help.\n"
        "- Keep replies short (2–4 sentences) and polite.\n"
        "- Do not greet repeatedly; greet the patient only once at the start of the visit."
    )

	user_prompt = ( # describing user prompt
        f"Discharge summary (for context):\n{summary}\n\n"
        f"Patient message:\n{message}\n\n"
        "Write a reply as the receptionist, following the rules above."
    )

	response = call_groq_chat( # calling llm which is hosted in groq
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model="openai/gpt-oss-20b",
        temperature=0.2,
        max_tokens=300,
    )

	return response, state, False



# if __name__ == "__main__": #for confirming the working condition of bot
#     state: State = {}
#     print("Receptionist test chat. Type 'exit' to quit.\n")

#     while True:
#         msg = input("You: ")
#         if msg.lower() in {"exit", "quit"}:
#             break

#         reply, state, handoff = receptionist_agent(msg, state)
#         print(f"Receptionist: {reply}")
#         print(f"(handoff_to_clinical={handoff})")
#         print("---")
