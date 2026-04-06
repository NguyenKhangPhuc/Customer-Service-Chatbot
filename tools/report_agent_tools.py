from fastapi import FastAPI
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from typing import Literal
from typing_extensions import NotRequired

from langchain.agents import AgentState
from langchain.messages import  ToolMessage
from langchain.tools import tool, ToolRuntime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from utils.config import *
from dotenv import load_dotenv
load_dotenv()

smtp_pass = os.getenv('SMTP_PASS')
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
    print(f"This is query for finding receiver email {full_problem_information}")
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
    print(f"Thiss iss context after semantic search {context}")
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
def go_back_to_subject_title(runtime: ToolRuntime[None, SupportState]) -> Command:
    """Go back to subject title step."""
    return Command(update={
        "messages": [
                        ToolMessage(
                        content=f"Go back to subject title state",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
        "current_step": "subject_title"
        })

@tool
def go_back_to_sender_email_step(runtime: ToolRuntime[None, SupportState]) -> Command:
    """Go back to sender email step step."""
    return Command(update={
        "messages": [
                        ToolMessage(
                        content=f"Go back to sender email state",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
        "current_step": "sender_email"
        })

@tool
def go_back_to_mail_description_step(runtime: ToolRuntime[None, SupportState]) -> Command:
    """Go back to the mail description/content step to edit the report body.""" 
    return Command(update={
        "messages": [
                        ToolMessage(
                        content=f"Go back to mail description state",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
        "current_step": "mail_description"
        })


@tool
def sending_email(receiver_email: str, sender_email: str, mail_subject: str, mail_description) -> str:
    """Sends a real email using Gmail SMTP and App Password."""
    

    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    GMAIL_APP_PASSWORD = smtp_pass 

    try:
        # 1. Tạo cấu trúc Email
        msg = MIMEMultipart()
        msg['From'] = receiver_email
        msg['To'] = sender_email
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