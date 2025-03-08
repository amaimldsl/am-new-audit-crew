from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool,FileReadTool,DirectoryReadTool
import os
import time
import logging
from typing import Dict, Any
from datetime import datetime
from litellm.exceptions import RateLimitError, APIError
import litellm
import json

# Set LiteLLM logging properly
os.environ['LITELLM_LOG'] = 'DEBUG'  # Replace deprecated set_verbose
os.environ["OTEL_SDK_DISABLED"] = "true"  # Add at the top of your crew.py
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pre_audit_crew.log')
    ]
)

# Add a custom exception at the top of the file
class ContentTooLargeError(Exception):
    pass

class EnhancedLLM(LLM):
    def __init__(self, *args, max_retries=999, base_delay=2, max_delay=61, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.logger = logging.getLogger(__name__)

    def call(self, *args, **kwargs):
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                return super().call(*args, **kwargs)
            except (APIError, litellm.exceptions.APIError) as e:
                if hasattr(e, 'status_code') and e.status_code == 413:
                    self.logger.error(f"Content too large, skipping website. Error: {str(e)}")
                    # Extract URL from context if available
                    current_url = kwargs.get('context', {}).get('website_url', 'unknown')
                    self.skipped_urls.add(current_url)
                    raise ContentTooLargeError(f"Skipped URL: {current_url}") from e
                # Existing error handling for other API errors
                retry_count += 1
                error_message = str(e)
                if "Expecting value: line 1 column 1 (char 0)" in error_message:
                    self.logger.warning(f"JSON parsing error detected (attempt {retry_count})")
                self.logger.error(f"API error on attempt {retry_count}: {error_message}")
                if retry_count >= self.max_retries:
                    self.logger.error("Max retries reached, raising error")
                    raise
                delay = min(self.base_delay * (2 ** retry_count), self.max_delay)
                self.logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            except Exception as e:
                self.logger.error(f"Unexpected error type {type(e)}: {str(e)}")
                raise


def handle_api_errors(func):
    def wrapper(*args, **kwargs):
        max_retries = 999
        base_delay = 15  # Increased base delay
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (APIError, Exception) as e:
                # Handle Deepseek's JSON parsing errors
                if "Expecting value: line 1 column 1 (char 0)" in str(e):
                    delay = base_delay * (2 ** attempt)  # Stronger backoff
                    logging.warning(f"Deepseek JSON Error (attempt {attempt+1}): Retrying in {delay}s")
                    time.sleep(delay)
                elif isinstance(e, APIError):
                    logging.warning(f"API Error (attempt {attempt+1}): {str(e)}")
                    time.sleep(10)
                else:
                    logging.error(f"Unexpected error: {type(e)} - {str(e)}")
                    raise
                
                if attempt == max_retries - 1:
                    logging.error("Max retries reached. Aborting.")
                    raise
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                raise
        return None
    return wrapper


# DeepSeek LLM configuration (unchanged)
#class DeepseekLLM(LLM):
#    def __init__(self, api_key, base_url, model):
#        super().__init__(
#            model=model,
#            api_key=api_key,
#            base_url=base_url,
#            temperature=0.3,
#            max_retries=99,
#            timeout=120,
##            max_tokens=4000
#        )

@CrewBase
class PreAuditCrew():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'


    def __init__(self, serper_api_key: str, topic: str, deepseek_api_key: str, deepseek_url: str, deepseek_model: str):
        self.skipped_urls = set()  # Track URLs causing errors
        self.llm = EnhancedLLM(
            model=deepseek_model,
            api_key=deepseek_api_key,
            base_url=deepseek_url,
            temperature=0.3,
            max_tokens=4000,
            max_retries=999,  # High retry count for persistence
            base_delay=2,
            max_delay=61
        )
        self.crew_llm = self.llm  # Use same LLM instance for crew
        self.topic = topic
        self.serper_api_key = serper_api_key
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        
        
        #self.directory_tool = DirectoryReadTool(directory='./search_results', name="DirectoryTool" )
        #self.file_tool = FileReadTool(name="FileTool" )
    
    #@handle_api_errors
    #def process_inputs(self, inputs: Dict[str, Any]) -> None:
    #    """Process inputs received from main.py"""
    #    self.topic = inputs.get('topic')
    #    self.serper_api_key = inputs.get('serper_api_key')
    #    if not self.topic or not self.serper_api_key:
    #        raise ValueError("Missing required inputs: topic and serper_api_key must be provided")

    
    ########################################################
    #def create_tools(self) -> list:
    #    from pydantic import PrivateAttr

     #   class SafeScraper(ScrapeWebsiteTool):
     #       _crew: Any = PrivateAttr()
     #       
     #       def __init__(self, crew, **kwargs):
     #           super().__init__(**kwargs)
     #           self._crew = crew

     #       def _run(self, website_url):
     #           if website_url in self._crew.skipped_urls:
     #               return f"Skipped URL {website_url}"
     #           try:
     #               return super()._run(website_url)
     #           except Exception as e:
     #               # Only skip for specific errors
     #               if "413" in str(e) or "Too Large" in str(e):
     #                   self._crew.skipped_urls.add(website_url)
     #               raise  # Re-raise to allow agent retry logic
     ##############################################################
     # 
     # 
    
    
    def create_tools(self) -> list:
        from crewai_tools import ScrapeWebsiteTool

        class SafeScraper(ScrapeWebsiteTool):
            def __init__(self, crew, **kwargs):
                super().__init__(**kwargs)
                self._crew = crew

            def _run(self, website_url: str, *args, **kwargs) -> str:  # Add *args and **kwargs
                """Handle website scraping with proper parameter signature"""
                if website_url in self._crew.skipped_urls:
                    return f"Skipped URL {website_url}"
                
                try:
                    # Explicitly pass only website_url to parent
                    content = super()._run(website_url=website_url)
                    if len(content) > 3500:
                        self._crew.skipped_urls.add(website_url)
                        raise ContentTooLargeError(f"Content too large: {website_url}")
                    return content
                except Exception as e:
                    if "413" in str(e) or "Too Large" in str(e):
                        self._crew.skipped_urls.add(website_url)
                    raise
                    
        return [
            SerperDevTool(api_key=self.serper_api_key),
            SafeScraper(
                crew=self,
                config={
                    "llm": self.llm,
                    "max_retries": 1,
                    "suppress_errors": False,
                    "ssl_verify": False
                }
            ),
            DirectoryReadTool(directory='./search_results'),
            FileReadTool(),
        ]



    #@handle_api_errors
    def get_agent_config(self, agent_name: str) -> dict:
        """Get agent configuration with topic substitution"""
        config = self.agents_config[agent_name]
        if isinstance(config, dict):
            return {k: v.format(topic=self.topic) if isinstance(v, str) else v 
                   for k, v in config.items()}
        return config

    #@handle_api_errors
    def get_task_config(self, task_name: str) -> dict:
        """Get task configuration with topic and date substitution"""
        config = self.tasks_config[task_name]
        if isinstance(config, dict):
            return {k: v.format(topic=self.topic, current_date=self.current_date) 
                   if isinstance(v, str) else v 
                   for k, v in config.items()}
        return config

    @agent
    #@handle_api_errors
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
    #@handle_api_errors
    def sub_processes_researcher(self) -> Agent:
        return Agent(  # Fixed: 4-space indentation
            config=self.get_agent_config('sub_processes_researcher'),
            verbose=True,
            llm=self.llm,
            tools=self.create_tools(),
            memory=True,
            max_retries=2,
            retry_delay=15
        )
        

        
    

            
       

    @agent
    #@handle_api_errors
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
    #@handle_api_errors
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
    #@handle_api_errors
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
                return self.sub_processes_researcher.execute(context=context)
            except ContentTooLargeError as e:
                current_url = str(e).split("Skipped URL: ")[-1]
                self.skipped_urls.add(current_url)
                logging.warning(f"Permanently skipped: {current_url}")
                return f"Skipped website: {current_url}"

        return Task(
            config=self.get_task_config('sub_processes_research_task'),
            llm=self.llm,
            agent=self.sub_processes_researcher(),
            tools=self.create_tools(),
            #expected_output="List of all sub-processes",
            #output_file='search_results/sub_processes.md',
            execution_fn=_custom_execute,
        )

    @task
    #@handle_api_errors
    def global_regulations_research_task(self) -> Task:
        def _custom_execute(context=None):
            try:
                return self.global_regulations_researcher.execute(context=context)
            except ContentTooLargeError as e:
                current_url = str(e).split("Skipped URL: ")[-1]
                self.skipped_urls.add(current_url)
                logging.warning(f"Permanently skipped: {current_url}")
                return f"Skipped website: {current_url}"

        return Task(
            config=self.get_task_config('global_regulations_research_task'),
            llm=self.llm,
            agent=self.global_regulations_researcher(),
            tools=self.create_tools(),
            #expected_output="List of all sub-processes",
            #output_file='search_results/global_regulations.md',
            execution_fn=_custom_execute,
        )

    
    

    @task
    #@handle_api_errors
    def uae_regulations_research_task(self) -> Task:
        def _custom_execute(context=None):
            try:
                return self.uae_regulations_researcher.execute(context=context)
            except ContentTooLargeError as e:
                current_url = str(e).split("Skipped URL: ")[-1]
                self.skipped_urls.add(current_url)
                logging.warning(f"Permanently skipped: {current_url}")
                return f"Skipped website: {current_url}"

        return Task(
            config=self.get_task_config('uae_regulations_research_task'),
            llm=self.llm,
            agent=self.uae_regulations_researcher(),
            tools=self.create_tools(),
            #expected_output="List of all sub-processes",
            #output_file='search_results/sub_processes.md',
            execution_fn=_custom_execute,
        )

    
    



    @task
    #@handle_api_errors
    def standards_research_task(self) -> Task:
        def _custom_execute(context=None):
            try:
                return self.standards_researcher.execute(context=context)
            except ContentTooLargeError as e:
                current_url = str(e).split("Skipped URL: ")[-1]
                self.skipped_urls.add(current_url)
                logging.warning(f"Permanently skipped: {current_url}")
                return f"Skipped website: {current_url}"

        return Task(
            config=self.get_task_config('standards_research_task'),
            llm=self.llm,
            agent=self.standards_researcher(),
            tools=self.create_tools(),
            #expected_output="List of all sub-processes",
            #output_file='search_results/sub_processes.md',
            execution_fn=_custom_execute,
        )



    @task
    def risk_research_task(self) -> Task:
        def _custom_execute(context=None):
            try:
                return self.risk_researcher.execute(context=context)  # Fixed agent reference
            except ContentTooLargeError as e:
                current_url = str(e).split("Skipped URL: ")[-1]
                self.skipped_urls.add(current_url)
                logging.warning(f"Permanently skipped: {current_url}")
                return f"Skipped website: {current_url}"

        return Task(
            config=self.get_task_config('risk_research_task'),
            llm=self.llm,
            agent=self.risk_researcher(),  # Correct agent
            tools=self.create_tools(),
            execution_fn=_custom_execute,
        )

          
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
            #output_file='search_results/PRCT.md',
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
            #output_file='Pre_Audit_Report.md',
            async_execution=False,
            description="Compile all findings from search_results folder into report. Preserve original content verbatim.",
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

    def kickoff(self, inputs: Dict[str, Any]) -> None:
        self.skipped_urls = set()  # Reset skipped URLs
        try:
            crew = self.crew()
            return crew.kickoff()
        except ContentTooLargeError as e:
            logging.error(f"Skipping website due to size: {str(e)}")
            return {"status": "partial_completion", "skipped_urls": [str(e)]}