#!/usr/bin/env python
import sys
import os
import warnings
from dotenv import load_dotenv
from crew import PreAuditCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew.
    """
    audit_subject = input("What is the audit subject? : ")
    #audit_subject = "Data Governance"

    # Load environment variables from .env file
    load_dotenv()

    # Access environment variables
    serper_api_key = os.getenv("SERPER_API_KEY")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    deepseek_url = os.getenv("DEEPSEEK_API_BASE")
    deepseek_model = os.getenv("DEEPSEEK_MODEL")

    inputs = {
        'topic': audit_subject,
        'serper_api_key': serper_api_key,
        'deepseek_api_key': deepseek_api_key,
        'deepseek_url': deepseek_url,
        'deepseek_model': deepseek_model
    }

    # Initialize and run the crew
    PreAuditCrew(
        topic = audit_subject,
        serper_api_key=serper_api_key,
        deepseek_api_key=deepseek_api_key,
        deepseek_url=deepseek_url,
        deepseek_model=deepseek_model
    ).crew().kickoff(inputs=inputs)

run()