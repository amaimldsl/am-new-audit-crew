from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, FileReadTool, DirectoryReadTool
import os
import time
import logging
import re
import string
from typing import Dict, Any, Set
from datetime import datetime
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

# Import enhanced components
from tools.enhanced_tools import (
    EnhancedLanguageDetection,
    WebsiteTrackingManager,
    create_enhanced_scrape_tool,  # Add this import
    EnhancedLLM,
    ContentTooLargeError,
    NonEnglishContentError
)

# Set LiteLLM logging properly
os.environ['LITELLM_LOG'] = 'DEBUG'
os.environ["OTEL_SDK_DISABLED"] = "true"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pre_audit_crew.log')
    ]
)

def pre_process_website_content(url, content, max_length=5000):
    """Pre-process website content to make it more manageable"""
    # If content is too long, truncate it
    if len(content) > max_length:
        try:
            # Use BeautifulSoup to extract the main content
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script, style, meta and other non-content tags
            for tag in soup(['script', 'style', 'meta', 'head', 'footer', 'nav']):
                tag.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Check if the content is mainly English
            detector = EnhancedLanguageDetection()
            if not detector.is_english(text):
                logging.warning(f"Content from {url} appears to be non-English, skipping")
                return "⚠️ This website contains primarily non-English content and has been skipped."
            
            # Clean up whitespace
            text = ' '.join(text.split())
            
            # Truncate if still too long
            if len(text) > max_length:
                text = text[:max_length] + "... [content truncated]"
            
            return text
        except Exception as e:
            logging.error(f"Error processing content from {url}: {e}")
            # If processing fails, just truncate
            return content[:max_length] + "... [content truncated]"
    
    return content

@CrewBase
class PreAuditCrew():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, serper_api_key: str, topic: str, deepseek_api_key: str, deepseek_url: str, deepseek_model: str):
        # Initialize website tracker manager
        self.tracker_manager = WebsiteTrackingManager()
        logging.info(f"Initialized with {self.tracker_manager.get_error_summary()}")
        
        # Initialize language detector
        self.language_detector = EnhancedLanguageDetection()
        
        self.skipped_urls = set()  # Track URLs causing errors in this session
        self.llm = EnhancedLLM(
            model=deepseek_model,
            api_key=deepseek_api_key,
            base_url=deepseek_url,
            temperature=0.2,  # Lower temperature for more factual responses
            max_tokens=4000,
            max_retries=5,
            base_delay=2,
            max_delay=61,
            tracker_manager=self.tracker_manager
        )
        self.crew_llm = self.llm  # Use same LLM instance for crew
        self.topic = topic
        self.serper_api_key = serper_api_key
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create output directories if they don't exist
        os.makedirs('./search_results', exist_ok=True)
    
    
    
    



    def create_tools(self) -> list:
        """Create the tools needed by agents for research using enhanced tools."""
        return [
            SerperDevTool(api_key=self.serper_api_key),
            create_enhanced_scrape_tool(
                tracker_manager=self.tracker_manager,
                language_detector=self.language_detector,
                config={
                    "llm": self.llm,
                    "max_retries": 2,
                    "suppress_errors": True,
                    "ssl_verify": False,
                    "timeout": 15
                }
            ),
            DirectoryReadTool(directory='./search_results'),
            FileReadTool(),
        ]



    def get_agent_config(self, agent_name: str) -> dict:
        """Get agent configuration with topic substitution"""
        config = self.agents_config[agent_name]
        if isinstance(config, dict):
            return {k: v.format(topic=self.topic) if isinstance(v, str) else v 
                   for k, v in config.items()}
        return config

    def get_task_config(self, task_name: str) -> dict:
        """Get task configuration with topic and date substitution"""
        config = self.tasks_config[task_name]
        if isinstance(config, dict):
            return {k: v.format(topic=self.topic, current_date=self.current_date) 
                   if isinstance(v, str) else v 
                   for k, v in config.items()}
        return config

    @agent
    def global_regulations_researcher(self) -> Agent:
        return Agent(   
            config=self.get_agent_config('global_regulations_researcher'),
            verbose=True,
            llm=self.llm,
            tools=self.create_tools(),
            memory=True,
            max_retries=2,
            retry_delay=15
        )
    
    @agent
    def sub_processes_researcher(self) -> Agent:
        return Agent(
            config=self.get_agent_config('sub_processes_researcher'),
            verbose=True,
            llm=self.llm,
            tools=self.create_tools(),
            memory=True,
            max_retries=2,
            retry_delay=15
        )

    @agent
    def uae_regulations_researcher(self) -> Agent:
        return Agent(
            config=self.get_agent_config('uae_regulations_researcher'),
            verbose=True,
            llm=self.llm,
            tools=self.create_tools(),
            memory=True,
            max_retries=2,
            retry_delay=15
        )

    @agent
    def standards_researcher(self) -> Agent:
        return Agent(
            config=self.get_agent_config('standards_researcher'),
            verbose=True,
            llm=self.llm,
            tools=self.create_tools(),
            memory=True,
            max_retries=2,
            retry_delay=15
        )

    @agent
    def risk_researcher(self) -> Agent:
        return Agent(
            config=self.get_agent_config('risk_researcher'),
            verbose=True,
            llm=self.llm,
            tools=self.create_tools(),
            memory=True,
            max_retries=2,
            retry_delay=15
        )
    
    @agent
    def prct_compilation_agent(self) -> Agent:
        return Agent(
            config=self.get_agent_config('prct_compilation_agent'),
            verbose=True,
            llm=self.llm,
            tools=self.create_tools(),
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.get_agent_config('reporting_analyst'),
            tools=self.create_tools(),
            verbose=True,
            llm=self.llm,
        )
    
    @task
    def sub_processes_research_task(self) -> Task:
        def _custom_execute(context=None):
            try:
                # Execute normally
                result = self.sub_processes_researcher.execute(context=context)
                return result
            except ContentTooLargeError as e:
                current_url = str(e).split("Content too large for LLM processing: ")[-1]
                self.skipped_urls.add(current_url)
                logging.warning(f"Permanently skipped and blocked: {current_url}")
                return f"⚠️ Skipped website due to size: {current_url}"
            except NonEnglishContentError as e:
                url = str(e).split("Non-English content detected: ")[-1]
                self.skipped_urls.add(url)
                self.tracker_manager.add_blocked_website(url, "non-english")
                logging.warning(f"Blocked non-English website: {url}")
                return f"⚠️ Skipped non-English website: {url}"
            except Exception as e:
                logging.error(f"Error in sub_processes_research_task: {str(e)}")
                if "413" in str(e) or "too large" in str(e).lower():
                    self.tracker_manager.add_blocked_website("unknown-from-error", "size-error")
                raise

        return Task(
            config=self.get_task_config('sub_processes_research_task'),
            llm=self.llm,
            agent=self.sub_processes_researcher(),
            tools=self.create_tools(),
            execution_fn=_custom_execute,
        )

    @task
    def global_regulations_research_task(self) -> Task:
        def _custom_execute(context=None):
            try:
                # Execute normally
                result = self.global_regulations_researcher.execute(context=context)
                return result
            except ContentTooLargeError as e:
                current_url = str(e).split("Content too large for LLM processing: ")[-1]
                self.skipped_urls.add(current_url)
                logging.warning(f"Permanently skipped and blocked: {current_url}")
                return f"⚠️ Skipped website due to size: {current_url}"
            except NonEnglishContentError as e:
                url = str(e).split("Non-English content detected: ")[-1]
                self.skipped_urls.add(url)
                self.tracker_manager.add_blocked_website(url, "non-english")
                logging.warning(f"Blocked non-English website: {url}")
                return f"⚠️ Skipped non-English website: {url}"
            except Exception as e:
                logging.error(f"Error in global_regulations_research_task: {str(e)}")
                if "413" in str(e) or "too large" in str(e).lower():
                    self.tracker_manager.add_blocked_website("unknown-from-error", "size-error")
                raise

        return Task(
            config=self.get_task_config('global_regulations_research_task'),
            llm=self.llm,
            agent=self.global_regulations_researcher(),
            tools=self.create_tools(),
            execution_fn=_custom_execute,
        )

    @task
    def uae_regulations_research_task(self) -> Task:
        def _custom_execute(context=None):
            try:
                # Execute normally
                result = self.uae_regulations_researcher.execute(context=context)
                return result
            except ContentTooLargeError as e:
                current_url = str(e).split("Content too large for LLM processing: ")[-1]
                self.skipped_urls.add(current_url)
                self.tracker_manager.add_blocked_website(current_url, "content-too-large")
                logging.warning(f"Permanently skipped and blocked: {current_url}")
                return f"⚠️ Skipped website due to size: {current_url}"
            except NonEnglishContentError as e:
                url = str(e).split("Non-English content detected: ")[-1]
                self.skipped_urls.add(url)
                self.tracker_manager.add_blocked_website(url, "non-english")
                logging.warning(f"Blocked non-English website: {url}")
                return f"⚠️ Skipped non-English website: {url}"
            except Exception as e:
                logging.error(f"Error in uae_regulations_research_task: {str(e)}")
                if "413" in str(e) or "too large" in str(e).lower():
                    self.tracker_manager.add_blocked_website("unknown-from-error", "size-error")
                raise

        return Task(
            config=self.get_task_config('uae_regulations_research_task'),
            llm=self.llm,
            agent=self.uae_regulations_researcher(),
            tools=self.create_tools(),
            execution_fn=_custom_execute,
        )

    @task
    def standards_research_task(self) -> Task:
        def _custom_execute(context=None):
            try:
                # Execute normally
                result = self.standards_researcher.execute(context=context)
                return result
            except ContentTooLargeError as e:
                current_url = str(e).split("Content too large for LLM processing: ")[-1]
                self.skipped_urls.add(current_url)
                self.tracker_manager.add_blocked_website(current_url, "content-too-large")
                logging.warning(f"Permanently skipped and blocked: {current_url}")
                return f"⚠️ Skipped website due to size: {current_url}"
            except NonEnglishContentError as e:
                url = str(e).split("Non-English content detected: ")[-1]
                self.skipped_urls.add(url)
                self.tracker_manager.add_blocked_website(url, "non-english")
                logging.warning(f"Blocked non-English website: {url}")
                return f"⚠️ Skipped non-English website: {url}"
            except Exception as e:
                logging.error(f"Error in standards_research_task: {str(e)}")
                if "413" in str(e) or "too large" in str(e).lower():
                    self.tracker_manager.add_blocked_website("unknown-from-error", "size-error")
                raise

        return Task(
            config=self.get_task_config('standards_research_task'),
            llm=self.llm,
            agent=self.standards_researcher(),
            tools=self.create_tools(),
            execution_fn=_custom_execute,
        )

    @task
    def risk_research_task(self) -> Task:
        def _custom_execute(context=None):
            try:
                # Execute normally
                result = self.risk_researcher.execute(context=context)
                return result
            except ContentTooLargeError as e:
                current_url = str(e).split("Content too large for LLM processing: ")[-1]
                self.skipped_urls.add(current_url)
                self.tracker_manager.add_blocked_website(current_url, "content-too-large")
                logging.warning(f"Permanently skipped and blocked: {current_url}")
                return f"⚠️ Skipped website due to size: {current_url}"
            except NonEnglishContentError as e:
                url = str(e).split("Non-English content detected: ")[-1]
                self.skipped_urls.add(url)
                self.tracker_manager.add_blocked_website(url, "non-english")
                logging.warning(f"Blocked non-English website: {url}")
                return f"⚠️ Skipped non-English website: {url}"
            except Exception as e:
                logging.error(f"Error in risk_research_task: {str(e)}")
                if "413" in str(e) or "too large" in str(e).lower():
                    self.tracker_manager.add_blocked_website("unknown-from-error", "size-error")
                raise

        return Task(
            config=self.get_task_config('risk_research_task'),
            llm=self.llm,
            agent=self.risk_researcher(),
            tools=self.create_tools(),
            execution_fn=_custom_execute,
        )
    
    def _extract_prct_entries(self, content):
        """Extract PRCT matrix entries from content."""
        entries = []
        
        # Look for PRCT entries in the content
        prct_sections = re.split(r'(?:Process|PRCT Matrix)[\s]*:', content)
        
        for section in prct_sections[1:]:  # Skip the first section which is before any PRCT
            try:
                process = re.search(r'(.+?)(?:Risk|$)', section, re.DOTALL)
                risk = re.search(r'Risk\s*:\s*(.+?)(?:Risk Rating|Control|Test|$)', section, re.DOTALL)
                risk_rating = re.search(r'Risk Rating\s*:\s*(.+?)(?:Control|Test|$)', section, re.DOTALL)
                control = re.search(r'Control\s*:\s*(.+?)(?:Test|$)', section, re.DOTALL)
                test = re.search(r'Test\s*:\s*(.+?)(?:$|Process)', section, re.DOTALL)
                
                # Extract source URL if available
                source_match = re.search(r'URL:\s*(https?://[^\s]+)', section)
                
                entry = {}
                
                if process:
                    entry["Process"] = process.group(1).strip()
                if risk:
                    entry["Risk"] = risk.group(1).strip()
                if risk_rating:
                    rating_text = risk_rating.group(1).strip()
                    # Normalize ratings to High/Medium/Low
                    if "high" in rating_text.lower():
                        entry["Risk Rating"] = "High"
                    elif "medium" in rating_text.lower() or "moderate" in rating_text.lower():
                        entry["Risk Rating"] = "Medium"
                    else:
                        entry["Risk Rating"] = "Low"
                if control:
                    entry["Control"] = control.group(1).strip()
                if test:
                    entry["Test"] = test.group(1).strip()
                if source_match:
                    entry["Source"] = source_match.group(1).strip()
                
                if entry:  # Only add non-empty entries
                    entries.append(entry)
            except Exception as e:
                logging.error(f"Error parsing PRCT section: {str(e)}")
                
        return entries

    @task
    def prct_compilation_task(self) -> Task:
        return Task(
            config=self.get_task_config('prct_compilation_task'),
            llm=self.llm,
            agent=self.prct_compilation_agent(),
            tools=self.create_tools(),
            context=[
                self.sub_processes_research_task(),
                self.global_regulations_research_task(),
                self.uae_regulations_research_task(),
                self.standards_research_task(),
                self.risk_research_task()
            ],
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.get_task_config('reporting_task'),
            llm=self.llm,
            agent=self.reporting_analyst(),
            tools=self.create_tools(),
            context=[
                self.sub_processes_research_task(),
                self.global_regulations_research_task(),
                self.uae_regulations_research_task(),
                self.standards_research_task(),
                self.risk_research_task(),
                self.prct_compilation_task()
            ],
            async_execution=False,
            description="Compile all findings with focus on top risk points in each category",
            callback=lambda output: logging.info(f"Report generated at: {output}")
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            manager_llm=self.crew_llm,
        )

    def kickoff(self, inputs: Dict[str, Any] = None) -> None:
        try:
            # Log blocked websites at the start
            logging.info(f"Starting with {self.tracker_manager.get_error_summary()}")
            
            # Run the crew
            crew = self.crew()
            result = crew.kickoff()
            
            # Get LLM statistics if available
            if hasattr(self.llm, 'get_stats'):
                llm_stats = self.llm.get_stats()
                logging.info(f"LLM statistics: {llm_stats}")
            
            # Log final status of blocked websites
            logging.info(f"Finished with {self.tracker_manager.get_error_summary()}")
            
            return result
        except ContentTooLargeError as e:
            current_url = str(e).split("Content too large for LLM processing: ")[-1]
            self.tracker_manager.add_blocked_website(current_url, "content-too-large")
            logging.error(f"Skipping website due to size: {str(e)}")
            return {"status": "partial_completion", "skipped_urls": [current_url]}
        except Exception as e:
            logging.error(f"Error during execution: {str(e)}")
            return {"status": "error", "message": str(e)}