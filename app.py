import uvicorn
from fastapi import FastAPI, Request
from src.graphs.graph_builder import GraphBuilder
from src.llms.groqllm import GroqLLM

import os
from dotenv import load_dotenv
try:
    load_dotenv()
except PermissionError:
    # In some sandboxed runners, reading `.env` can be disallowed.
    # The app can still run if env vars are provided by the process environment.
    pass

app=FastAPI()

print(os.getenv("LANGCHAIN_API_KEY"))

_langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if _langchain_api_key:
    os.environ["LANGSMITH_API_KEY"] = _langchain_api_key

## API's

@app.post("/blogs")
async def create_blogs(request:Request):
    
    data=await request.json()
    topic= data.get("topic","")
    language = data.get("language", '')
    print(language)

    ## get the llm object

    groqllm=GroqLLM()
    llm=groqllm.get_llm()

    ## get the graph
    graph_builder=GraphBuilder(llm)
    if topic and language:
        graph=graph_builder.setup_graph(usecase="language")
        state=graph.invoke({"topic":topic,"current_language":language.lower()})

    elif topic:
        graph=graph_builder.setup_graph(usecase="topic")
        state=graph.invoke({"topic":topic})
    

    return {"data":state}

if __name__=="__main__":
    uvicorn.run("app:app",host="0.0.0.0",port=8000,reload=True)

