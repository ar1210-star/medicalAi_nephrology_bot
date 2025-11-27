import os
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_groq_chat(system_prompt: str,user_prompt: str, model: str="openai/gpt-oss-20b", temperature: float=0.4,max_tokens: int = 300) -> str:
    """call Groq chat completion API."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
          temperature=temperature,
          max_tokens=max_tokens
    )
    
    return resp.choices[0].message.content