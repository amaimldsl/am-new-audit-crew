#!/usr/bin/env python
import sys
import os
import warnings
import logging
import time
from dotenv import load_dotenv
from crew import PreAuditCrew
from datetime import datetime

# Suppress all warnings - helps with regex and pydantic issues
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'pre_audit_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

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

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import langdetect
        logging.info("langdetect package is available")
    except ImportError:
        logging.warning("langdetect package is not installed. Language detection will use fallback method.")
        logging.warning("Install with: pip install langdetect")
    
    try:
        import chardet
        logging.info("chardet package is available")
    except ImportError:
        logging.warning("chardet package is not installed. Encoding detection may be less accurate.")
        logging.warning("Install with: pip install chardet")

def run():
    """
    Run the enhanced pre-audit crew.
    """
    print("\n========== ENHANCED PRE-AUDIT RESEARCH TOOL ==========")
    print("This tool will research and compile information about your audit subject from reliable sources.\n")
    
    # Check for required dependencies
    check_dependencies()

    # Get audit subject from user
    audit_subject = input("Enter the audit subject (e.g., 'Basel III Compliance', 'IFRS 9 Implementation'): ")
    #audit_subject = "Basel PI , PII , PIII compliance for Banks"

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
        
        # Display blocked websites info before starting
        if hasattr(crew, 'tracker_manager'):
            print(f"Note: {crew.tracker_manager.get_error_summary()['total_blocked_sites']} websites are blocked from previous runs")
        
        result = crew.crew().kickoff()
        
        # Calculate and display completion time
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n========== RESEARCH COMPLETED ==========")
        print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Report generated: Pre_Audit_Report.md")
        print(f"Additional files created in ./search_results/ directory")
        
        # Report on tracking statistics
        if hasattr(crew, 'tracker_manager'):
            error_summary = crew.tracker_manager.get_error_summary()
            session_stats = crew.tracker_manager.get_session_stats()
            
            print("\n========== SESSION STATISTICS ==========")
            print(f"Session ID: {session_stats['session_id']}")
            print(f"Errors encountered: {session_stats['error_count']}")
            print(f"Websites blocked in this session: {session_stats['blocked_count']}")
            print(f"Total blocked websites: {error_summary['total_blocked_sites']}")
            
            # LLM statistics if available
            if hasattr(crew.llm, 'get_stats'):
                llm_stats = crew.llm.get_stats()
                print("\n========== LLM STATISTICS ==========")
                print(f"Total calls: {llm_stats['total_calls']}")
                print(f"Success rate: {llm_stats['success_rate']}")
                if llm_stats['total_calls'] > 0:
                    print(f"Average retries: {float(llm_stats['average_retries']):.2f}")
                    if 'error_types' in llm_stats and llm_stats['error_types']:
                        print("Error types encountered:")
                        for error_type, count in llm_stats['error_types'].items():
                            print(f"  - {error_type}: {count}")
                    else:
                        print("No errors encountered")
            
        print("\nReview the generated report for critical points in each category.")
        
        return result
        
    except Exception as e:
        logging.error(f"Error running pre-audit crew: {str(e)}")
        print("\n========== ERROR ==========")
        print(f"An error occurred: {str(e)}")
        print("Check the log file for more details.")
        return None

if __name__ == "__main__":
    run()