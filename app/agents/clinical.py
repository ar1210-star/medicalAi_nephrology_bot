from typing import List, Dict, Any, Optional, Tuple
from app.llm.groq_client import call_groq_chat
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from app.tools.web_search import web_search

State = Dict[str, Any]
emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vect_store = Chroma(
    embedding_function=emb_model,
    persist_directory="chroma_db_v2/clinical-nephrology_db"
)
# retriver = vect_store.as_retriever(search_kwargs={"k":3})


def book_context(docs: List[Any],patient_record: Optional[Dict[str, Any]]=None) -> str:
    """ returns the context string from retrived documents.
    """
    lines = []
    
    if patient_record:
        pr = patient_record
        lines.append("=== Patient Discharge Summary ===")
        lines.append(f"Primary diagnosis: {pr.get('primary_diagnosis')}")
        lines.append(f"Discharge date: {pr.get('discharge_date')}")
        lines.append(f"Medications: {pr.get('medications')}")
        lines.append(f"Dietary restrictions: {pr.get('dietary_restrictions')}")
        lines.append(f"Follow-up: {pr.get('follow_up')}")
        lines.append(f"Warning signs: {pr.get('warning_signs')}")
        lines.append("")
    
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        page = meta.get("page")
        chunk_idx = meta.get("chunk_index")
        header = f"[Source {idx} | Page {page}, Chunk {chunk_idx}]"
        lines.append(header)
        lines.append(doc.page_content)
        lines.append("")

    return "\n".join(lines)
  

def web_context(results: List[Dict[str, Any]]) -> str:
    """ returns the context string from web search results.
    """
    lines = []
    for idx, r in enumerate(results, start=1):
        lines.append(
          f"[Web {idx}] {r.get('title')}\n"
          f"URL: {r.get('url')}\n"
          f"Snippet: {r.get('snippet')}\n"
        )
    
    return "\n\n".join(lines)
  
  
def wants_latest_or_web(question: str) -> bool:
    """ returns if the question asks for latest or noteworthy information.
    """
    text = question.lower()
    keywords = [
        "latest",
        "recent",
        "new research",
        "new study",
        "guideline",
        "2023",
        "2024",
        "2025",
        "web search",
        "check online",
    ]
    return any(k in text for k in keywords)
  
def clinical_agent(message: str,state: State, allow_web: bool = True) -> Tuple[str, State]:
      """ clinical agent to handle human queries related to diagnosis, searches web if they want to know latest info and suggest them to go to doctor. if they have any serious issues.
      """
      patient_record = state.get("patient_record")
      ask_for_web = wants_latest_or_web(message)
      
      docs = vect_store.similarity_search(message, k=6)
      
      if docs and not (ask_for_web and allow_web):
          context = book_context(docs, patient_record)
          
          system_prompt = (
            "You are a clinical nephrology assistant answering questions for a recently discharged patient.\n"
            "- Use only the context from the nephrology reference and discharge summary below.\n"
            "- Refer to the snippets using the [Source N] labels when needed.\n"
            "- If the context does not fully answer the question, say that clearly.\n"
            "- Keep the answer focused and easy to understand.\n"
            "- End with a short line reminding the user that this does not replace their doctor's advice.\n"
          )
          
          user_prompt = (
            f"Patient question:\n{message}\n\n"
            f"---\n"
            f"Context:\n{context}\n"
          )
          
          answer = call_groq_chat(
              system_prompt=system_prompt,  
              user_prompt=user_prompt,
              model="openai/gpt-oss-20b",
              temperature=0.1,
              max_tokens=400
          )
          
          return answer, state
      
      if allow_web:
          web_results = web_search(message, num_results=3)
          
          bc = book_context(docs, patient_record) if docs else ""
          wc = web_context(web_results) if web_results else ""
          
          if not book_context and not web_context:
              fallback = (
                "I could not find enough reliable information in the textbook or via web search "
                "to answer this question. Please discuss this directly with your doctor."
              )
              return fallback, state
          
          system_prompt = (
            "You are a clinical nephrology assistant.\n"
            "- You have textbook context (labelled [Source N]) and web context (labelled [Web N]).\n"
            "- Prefer textbook information when possible, but you may mention web sources for newer data.\n"
            "- If there is any conflict, say that the treating doctor should decide.\n"
            "- Keep the answer concise and clear.\n"
            "- End with a short reminder that this does not replace medical advice from their own doctor.\n"
          )
          
          user_prompt = (
            f"Patient question:\n{message}\n\n"
            f"---\n"
            f"Textbook context (may be empty):\n{bc}\n\n"
            f"---\n"
            f"Web search results (may be empty):\n{wc}\n"
          )
          
          answer = call_groq_chat(
              system_prompt=system_prompt,  
              user_prompt=user_prompt,
              model="openai/gpt-oss-20b",
              temperature=0.1,
              max_tokens=400
          )
          
          return answer, state
      fallback = (
        "I could not find enough reliable information in the textbook to answer this question. "
        "Please discuss this directly with your doctor."
      )
      return fallback, state

if __name__ == "__main__":
    state: State = {}
    print("Clinical agent test (RAG + optional web). Type 'exit' to quit.\n")

    while True:
        q = input("You: ")
        if q.lower() in {"exit", "quit"}:
            break

        reply, state = clinical_agent(q, state, allow_web=True)
        print(f"Clinical: {reply}")
        print("---")