
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import SummarizationMiddleware
from middleware.report_agent_middleware import apply_step_config
from tools.report_agent_tools import *
from utils.config import model


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
