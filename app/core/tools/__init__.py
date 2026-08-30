from app.core.tools.rag_tool import search_similar_cases
from app.core.tools.read_log_tool import read_log_snippet

TOOLS = [search_similar_cases, read_log_snippet]
