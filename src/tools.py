"""External tools for the RAG system"""
from langchain_community.tools.tavily_search import TavilySearchResults


# Web search tool
web_search_tool = TavilySearchResults(k=3)
