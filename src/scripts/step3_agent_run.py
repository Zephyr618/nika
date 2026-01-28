import argparse
import asyncio
import logging

from agent.react_agent import BasicReActAgent
from nika.utils.logger import system_logger
from nika.utils.session import Session


def _agent_selector(agent_type: str, llm_backend: str, model: str, max_steps: int = 20):
    match agent_type.lower():
        case "react":
            return BasicReActAgent(llm_backend=llm_backend, model=model, max_steps=max_steps)
        case _:
            pass


def start_agent(agent_type: str, llm_backend: str, model: str, max_steps: int):
    logger = system_logger

    session = Session()
    session.load_running_session()
    session.update_session("agent_type", agent_type)
    session.update_session("llm_backend", llm_backend)
    session.update_session("model", model)
    session.start_session()

    logger.info(f"Starting agent: {agent_type}  with backend {model} in session {session.session_id}")
    agent = _agent_selector(agent_type, llm_backend, model, max_steps=max_steps)
    asyncio.run(agent.run(task_description=session.task_description))

    # stop session
    session.end_session()
    logger.info(f"Agent run completed for session ID: {session.session_id}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description=(
            "Run the specified agent to start troubleshooting. \n"
            "Note: the backend LLM must be configured before running this command."
        )
    )

    parser.add_argument(
        "--agent_type",
        type=str,
        default="ReAct",
        help="Type of agent to run (default: ReAct)",
    )

    parser.add_argument(
        "--llm_backend",
        type=str,
        default="openai",
        help="LLM backend for the agent, options: openai (default), ollama, deepseek.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Backend model for the agent (default: gpt-5-mini)",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=20,
        help="Maximum steps for the agent to take (default: 20)",
    )

    args = parser.parse_args()
    start_agent(args.agent_type, args.llm_backend, args.model, args.max_steps)
