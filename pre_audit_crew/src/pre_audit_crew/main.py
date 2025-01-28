#!/usr/bin/env python
import sys
import os
import warnings
from dotenv import load_dotenv
from crew import PreAuditCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """

    #audit_subject = input("What is the audit subject? : ")
    audit_subject = "Data Governance"


    # Load environment variables from .env file
    load_dotenv()

    # Now you can access the API key
    serper_api_key = os.getenv("SERPER_API_KEY")

    inputs = {
        'topic': audit_subject,
        'serper_api_key' : serper_api_key,

    }
    PreAuditCrew().crew().kickoff(inputs=inputs)

run()