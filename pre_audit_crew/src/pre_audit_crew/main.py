#!/usr/bin/env python
import sys
import os
import warnings
import logging
import time
from dotenv import load_dotenv
from crew import PreAuditCrew
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'pre_audit_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def setup_environment():
    """Setup environment and validate required variables."""
    # Load environment variables from .env file
    load_dotenv()

    # Access environment variables with validation
    required_vars = [
        "SERPER_API_KEY", 
        "DEEPSEEK_API_KEY", 
        "DEEPSEEK_API_BASE", 
        "DEEPSEEK_MODEL"
    ]
    
    env_vars = {}
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            env_vars[var.lower()] = value
    
    if missing_vars:
        logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logging.error("Please ensure these are set in your .env file or environment")
        sys.exit(1)
        
    return env_vars

def create_output_directories():
    """Create necessary output directories if they don't exist."""
    directories = ['./search_results']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")

def run():
    """
    Run the enhanced pre-audit crew.
    """
    print("\n========== ENHANCED PRE-AUDIT RESEARCH TOOL ==========")
    print("This tool will research and compile the TOP 10 most critical")
    print("points for your audit subject from reliable sources.\n")

    # Get audit subject from user
    audit_subject = input("Enter the audit subject (e.g., 'Basel III Compliance', 'IFRS 9 Implementation'): ")
    
    if not audit_subject:
        logging.error("No audit subject provided. Exiting.")
        sys.exit(1)
    
    print(f"\nResearching: {audit_subject}")
    print("This process will take some time to complete as we search and analyze multiple sources.")
    print("You'll see logs of the progress as each step completes.\n")

    # Setup environment and validate variables
    env_vars = setup_environment()
    
    # Create necessary directories
    create_output_directories()
    
    # Start time for tracking
    start_time = time.time()
    
    try:
        # Initialize and run the crew
        crew = PreAuditCrew(
            topic=audit_subject,
            serper_api_key=env_vars['serper_api_key'],
            deepseek_api_key=env_vars['deepseek_api_key'],
            deepseek_url=env_vars['deepseek_api_base'],
            deepseek_model=env_vars['deepseek_model']
        )
        
        result = crew.kickoff(inputs={
            'topic': audit_subject,
            'serper_api_key': env_vars['serper_api_key']
        })
        
        # Calculate and display completion time
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n========== RESEARCH COMPLETED ==========")
        print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Report generated: Pre_Audit_Report.md")
        print(f"Additional files created in ./search_results/ directory")
        
        if hasattr(crew, 'skipped_urls') and crew.skipped_urls:
            print(f"\nNote: {len(crew.skipped_urls)} URLs were skipped due to content size limitations.")
            
        print("\nReview the generated report for the top 10 critical points in each category.")
        
        return result
        
    except Exception as e:
        logging.error(f"Error running pre-audit crew: {str(e)}")
        print("\n========== ERROR ==========")
        print(f"An error occurred: {str(e)}")
        print("Check the log file for more details.")
        return None

if __name__ == "__main__":
    run()