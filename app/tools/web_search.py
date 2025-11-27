import os
from typing import Any, Dict, List

from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv() #loading environment variables

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) # starting up tavily client with api

def web_search(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
	"""Perform a web search using Tavily API."""
	resp = client.search(query=query, num_results=num_results,include_raw_content=False,include_images=False,include_answer=False) #response by messsage
	
	results: List[Dict[str,Any]] = []
	for item in resp.get("results", []):
		result = { # appending results in below format
			"title": item.get("title", ""),
			"link": item.get("link", ""),
			"snippet": item.get("snippet", "")
		}
		results.append(result)
	return results


