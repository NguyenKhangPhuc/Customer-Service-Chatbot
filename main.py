from fastapi import FastAPI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client
from dotenv import load_dotenv
import uvicorn
from pydantic import BaseModel
from agents.report_agent import agent
from utils.config import *
load_dotenv()
app = FastAPI()


class ChatRequest(BaseModel):
    query: str
    user_id: str

class Intent(BaseModel):
    is_report_related: bool
    wants_to_cancel: bool
    is_ask_info: bool

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    config = {"configurable": {"thread_id": request.user_id}}
    intent_llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(Intent)
    print("What happen here")
    user_intent = intent_llm.invoke(f"""You are an Intent Classifier for a multi-agent system. 
Your task is to determine if the user wants to continue the EMAIL REPORT process or QUIT to ask for general information.

USER QUERY: "{request.query}"

CLASSIFICATION RULES:
1. **REPORT_RELATED**: 
   - User wants to change/edit information (subject, sender email, description).
   - User asks "What should I do next?" or "What's the next step?" in the report.
   - User provides the information requested by the report agent.
   - User confirms or denies the draft.
   - Basically, any action that keeps the reporting flow alive.

2. **GENERAL_QUERY**: 
   - IMPORTANT: ONLY QUIT IF THE USER TELL YOU TO QUIT OR THEY ASK ANY INFORMATION NOT RELATED TO REPORT TITLE, REPORT SENDER EMAIL, REPORT DESCRIPTION,

""", config=config)
    current_state = agent.get_state(config)
    if user_intent.wants_to_cancel:
        agent.update_state(config, {"current_step": "subject_title_retrieved", "subject_title": None})
        return {"reply": "I have quit from email sending step, what can I help you in informations?"}

    is_in_middle_of_email = current_state.values.get("current_step") is not None
    print(f"user_intent {user_intent}")
    if (user_intent.is_report_related or is_in_middle_of_email) and not user_intent.is_ask_info:
        response = agent.invoke({"messages": [{"role": "user", "content": request.query}]}, config=config)
        for msg in response["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    print(f"--- AI CALLS TOOL: {tool_call['name']} ---")
                    print(f"Args: {tool_call['args']}")
            
            if msg.type == "tool":
                print(f"--- TOOL {msg.name} RETURNS ---")
                print(f"Content: {msg.content}")
        return {"reply": response["messages"][-1].text}
        
    else:
        return similaritySearch(query=request.query)


def similaritySearch(query: str):
        query_vector = embeddings.embed_query(query)
        rpc_response = supabase.rpc(
                "match_documents", 
            {
                "query_embedding": query_vector,
                "match_count": 3,
                "filter": {}
            }
        ).execute()
        docs = rpc_response.data
        context = "\n".join([d['content'] for d in docs])
        
        llm = ChatOpenAI(model="gpt-4o")
        response = llm.invoke(f"Base on the above information: {context}. please answer briefly below 150 characters: {query}")
        return {"reply": response.content, "type": "semantic"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)