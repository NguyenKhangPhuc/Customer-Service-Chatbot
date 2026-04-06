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

from tools.report_agent_tools import confirm_final_email, description_retriever_tool, go_back_to_mail_description_step, go_back_to_sender_email_step, go_back_to_subject_title, receiver_email_retriever_tool, sender_email_retriever_tool, sending_email, subject_title_retriever_tool
from utils.prompts import *

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