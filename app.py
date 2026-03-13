import uvicorn
from fastapi import FastAPI, Request
from src.graphs.graph_builder import GraphBuilder
from src.llms.groqllm import GroqLLM

import os
from dotenv import load_dotenv
try:
    load_dotenv()
except PermissionError:
    pass

app=FastAPI()

print(os.getenv("LANGCHAIN_API_KEY"))

_langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if _langchain_api_key:
    os.environ["LANGSMITH_API_KEY"] = _langchain_api_key

## API
@app.post("/blogs")
async def create_blogs(request:Request):
    
    data=await request.json()
    topic= data.get("topic","")
    language = (data.get("language", "") or "").strip().lower()
    print(language)

    groqllm=GroqLLM()
    llm=groqllm.get_llm()

    graph_builder=GraphBuilder(llm)
    if topic and language and language != "english":
        graph=graph_builder.setup_graph(usecase="language")
        state=graph.invoke({"topic":topic,"current_language":language})

    elif topic:
        graph=graph_builder.setup_graph(usecase="topic")
        state=graph.invoke({"topic":topic})
        # Normalize response shape for clients (Postman)
        state["current_language"] = "english"

    blog = (state or {}).get("blog", {}) if isinstance(state, dict) else {}
    return {
        "title": blog.get("title", ""),
        "content": blog.get("content", ""),
        "language": (state or {}).get("current_language", language or "english") if isinstance(state, dict) else (language or "english"),
        "raw_state": state,
    }

if __name__=="__main__":
    uvicorn.run("app:app",host="0.0.0.0",port=8000,reload=True)
    

