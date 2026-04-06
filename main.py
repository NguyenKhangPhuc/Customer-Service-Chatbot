from fastapi import FastAPI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client
import os
from dotenv import load_dotenv
import uvicorn
from pydantic import BaseModel

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from typing import Callable, Literal
from typing_extensions import NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()
app = FastAPI()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
smtp_pass = os.getenv('SMTP_PASS')
supabase: Client = create_client(supabase_url, supabase_key)


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents" 
)

model = init_chat_model("gpt-4o")

SupportStep = Literal["subject_title", "sender_email", "mail_description", "receiver_email","final_confirmation"]

class SupportState(AgentState):
    """State for customer support workflow."""
    current_step: NotRequired[SupportStep]
    subject_title: NotRequired[str]
    sender_email: NotRequired[str]
    mail_description: NotRequired[str]
    receiver_email: NotRequired[str]
    final_confirmation: NotRequired[Literal['Yes','No']]
    send_email: NotRequired[str]

@tool
def subject_title_retriever_tool(
    subject_title: str,
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the subject or title of the support request."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Subject title recorded: {subject_title}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "subject_title": subject_title,
            "current_step": "sender_email",
        }
    )

@tool
def sender_email_retriever_tool(
    email: str,
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the sender email"""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Sender Email Retrieved: {email}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "sender_email": email,
            "current_step": "mail_description",
        }
    )

@tool
def description_retriever_tool(
    description: str,
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the description of the support request."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Email description: {description}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "mail_description": description,
        }
    )
@tool
def receiver_email_retriever_tool(
    full_problem_information: str, 
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Using above information to search for suitable receiver email in vector db"""
    query_vector = embeddings.embed_query(full_problem_information)

    rpc_response = supabase.rpc(
        "match_documents", 
        {
            "query_embedding": query_vector,
            "match_count": 2,
            "filter": {}
        }
    ).execute()

    docs = rpc_response.data
    context = "\n".join([d['content'] for d in docs])

    temp_llm = ChatOpenAI(model="gpt-4o-mini")
    email_extract = temp_llm.invoke(
        f"Base on the above information of the problem: {context}."
        "Please find the most suitable email for the problem: {full_problem_information}. "
        "Only return the email, nothing else!!"
    )
    
    receiver_email = email_extract.content.strip()

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Đã tìm thấy email người nhận phù hợp: {receiver_email}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "receiver_email": receiver_email,
            "current_step": "final_confirmation",
        }
    )

@tool
def confirm_final_email(
    confirmed: Literal['Yes','No'],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the confirmation decision of the user"""
    if confirmed == 'Yes':
        return Command(
            update={
                "messages": [
                        ToolMessage(
                        content=f"Confirmation from user: {confirmed}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "final_confirmation": "YES",
                "current_step": "final_confirmation", 
            }
        )
    else:
        # Khi là NO:
        return Command(
            update={
                "messages": [
                        ToolMessage(
                        content=f"Confirmation from user: {confirmed}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "final_confirmation": "NO",
                "current_step": "final_confirmation", 
            }
        )
    
@tool
def go_back_to_subject_title() -> Command:
    """Go back to subject title step."""
    return Command(update={"current_step": "subject_title"})

@tool
def go_back_to_sender_email_step() -> Command:
    """Go back to sender email step step."""
    return Command(update={"current_step": "sender_email"})

@tool
def go_back_to_mail_description_step() -> Command:
    """Go back to the mail description/content step to edit the report body.""" 
    return Command(update={"current_step": "mail_description"})


@tool
def sending_email(receiver_email: str, sender_email: str, mail_subject: str, mail_description) -> str:
    """Sends a real email using Gmail SMTP and App Password."""
    

    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    GMAIL_APP_PASSWORD = smtp_pass 

    try:
        # 1. Tạo cấu trúc Email
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = mail_subject
        
        # Thêm nội dung mail
        msg.attach(MIMEText(mail_description, 'plain'))

        # 2. Kết nối và gửi
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Bảo mật kết nối
        server.login(sender_email, GMAIL_APP_PASSWORD)
        
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()

        return f"Email sent successfully to {receiver_email}!"
    
    except Exception as e:
        return f"Failed to send email. Error: {str(e)}"

SUBJECT_TITLE_PROMPT = """You are a professional support assistant for "K P A P P" application.
CURRENT STEP: Collecting Subject Title

Your task:
1. Greet the customer and ask them for a concise subject/title for their support request.
2. If they provide an unrelated information about the question, kindly ask them again, do not use tool 'subject_title_retriever_tool'
until they provide a suitable information.
3. Once they provide a suitable topic, use 'subject_title_retriever_tool' to record it.

Guidelines:
- Keep it professional and brief.
- Do not ask for their email or description yet."""

SENDER_EMAIL_PROMPT = """You are a professional support assistant for KPAPP.
CURRENT STEP: Collecting Sender Email
CONTEXT: You are processing a request about: {subject_title}

Your task:
1. Acknowledge the subject provided.
2. Ask the customer for their contact email address so we can reach back to them.
3. If they provide an unrelated information about the question, kindly ask them again, do not use tool 'sender_email_retriever_tool'
until they provide a suitable information
4. Use 'sender_email_retriever_tool' to save the email once provided.

Guidelines:
- Ensure the input looks like a valid email address before calling the tool."""

MAIL_DESCRIPTION_PROMPT = """You are a professional support assistant for KPAPP.
CURRENT STEP: Collecting Issue Description & Automatic Department Routing
CONTEXT: Subject: {subject_title} | From: {sender_email}

Your task:
1. Ask the customer to provide a detailed description of the issue or the content they wish to send.
2. ONCE the user provides the description, you MUST perform these two actions IN ORDER:
   - First, call 'description_retriever_tool' to save the user's input.
   - Immediately after, call 'receiver_email_retriever_tool' to find the appropriate support email. 
     For this tool, combine the {subject_title} and the new description into a single search query.

Guidelines:
- DO NOT wait for a user response between calling the two tools. 
- After both tools have finished, you will automatically be moved to the final confirmation step to show the results to the user."""

FINAL_CONFIRMATION_PROMPT = """You are a professional support assistant for KPAPP.
CURRENT STEP: Final Review & Send

SUMMARY OF YOUR REPORT:
- From: {sender_email}
- To: {receiver_email}
- Subject: {subject_title}
- Content: {mail_description}

YOUR TASKS:
1. Present the full email draft clearly to the user.
2. Ask: "Is this information correct? Please say 'Yes' to send the email or 'No' if you need to edit something."

LOGIC:
- If user confirms (YES): Call 'confirm_final_email(is_confirmed=True)'. 
     (After calling the confirm_final_email with is_confirmed=True, 
     Execute the 'sending_email' tool immediately with email=email summary above).
    
- IF the user indicates any information is WRONG or wants to EDIT:
    Use the appropriate tool to return to the specific step:
    * To change the Subject: Use 'go_back_to_subject_title'
    * To change the Sender Email: Use 'go_back_to_sender_email_step'
    * To change the Description/Content: Use 'go_back_to_mail_description_step'
    
    After calling the tool, explain to the user that you are moving back to that step to make corrections."""

STEP_CONFIG = {
    "subject_title_retrieved": {
        "prompt": SUBJECT_TITLE_PROMPT,
        "tools": [subject_title_retriever_tool],
        "requires": [],
    },
    "sender_email": {
        "prompt": SENDER_EMAIL_PROMPT,
        "tools": [sender_email_retriever_tool],
        "requires": ["subject_title"], 
    },
    "mail_description": {
        "prompt": MAIL_DESCRIPTION_PROMPT,
        "tools": [description_retriever_tool, receiver_email_retriever_tool],
        "requires": ["subject_title", "sender_email"],
    },
    "final_confirmation": {
        "prompt": FINAL_CONFIRMATION_PROMPT,
        "tools": [confirm_final_email, sending_email, go_back_to_subject_title, go_back_to_sender_email_step, go_back_to_mail_description_step],
        "requires": ["subject_title", "sender_email", "mail_description", "receiver_email"],
    },
}


@wrap_model_call
def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Configure agent behavior based on the current step."""
    # Get current step (defaults to subject_title_retrieved for first interaction)
    current_step = request.state.get("current_step", "subject_title_retrieved")

    # Look up step configuration
    step_config = STEP_CONFIG[current_step]

    # Validate required state exists
    for key in step_config["requires"]:
        if request.state.get(key) is None:
            raise ValueError(f"{key} must be set before reaching {current_step}")

    # Format prompt with state values
    system_prompt = step_config["prompt"].format(**request.state)

    # Inject system prompt and step-specific tools
    request = request.override(
        system_prompt=system_prompt,
        tools=step_config["tools"],
    )

    return handler(request)


all_tools = [
    subject_title_retriever_tool,
    sender_email_retriever_tool,
    description_retriever_tool,
    receiver_email_retriever_tool,
    confirm_final_email,
    sending_email,
    go_back_to_mail_description_step,
    go_back_to_sender_email_step,
    go_back_to_subject_title
]


agent = create_agent(
    model,
    tools=all_tools,
    state_schema=SupportState,
    middleware=[
        apply_step_config,
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 10)
        )
    ],
    checkpointer=InMemorySaver(),
)

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
    user_intent = intent_llm.invoke(f"Analyze the user intention: {request.query}")
    current_state = agent.get_state(config)
    query_vector = embeddings.embed_query(request.query)
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
        query_vector = embeddings.embed_query(request.query)
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
        response = llm.invoke(f"Base on the above information: {context}. please answer briefly below 200 characters: {request.query}")
        return {"reply": response.content, "type": "semantic"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)