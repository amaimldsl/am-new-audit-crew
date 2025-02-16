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

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pre_audit_crew.log')
    ]
)

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
                retry_count += 1
                error_message = str(e)
                
                # Handle JSON decode errors specifically
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

    def create_tools(self) -> list:
        return [
            SerperDevTool(api_key=self.serper_api_key),
            ScrapeWebsiteTool(),
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
    def risk_researcher(self) -> Agent:
        return Agent(
            config=self.get_agent_config('risk_researcher'),
            verbose=True,
            llm=self.llm,
            tools=self.create_tools(),
            memory=True,
            max_retries=99,
            retry_delay=30
        )

    @agent
    #@handle_api_errors
    def sub_processes_researcher(self) -> Agent:
        return Agent(
            config=self.get_agent_config('sub_processes_researcher'),
            verbose=True,
            llm=self.llm,
            tools=self.create_tools(),
        )

    @agent
    #@handle_api_errors
    def global_regulations_researcher(self) -> Agent:
        return Agent(
            config=self.get_agent_config('global_regulations_researcher'),
            verbose=True,
            llm=self.llm,
            tools=self.create_tools(),
        )

    @agent
    #@handle_api_errors
    def uae_regulations_researcher(self) -> Agent:
        return Agent(
            config=self.get_agent_config('uae_regulations_researcher'),
            verbose=True,
            llm=self.llm,
            tools=self.create_tools(),
        )

    @agent
    #@handle_api_errors
    def standards_researcher(self) -> Agent:
        return Agent(
            config=self.get_agent_config('standards_researcher'),
            verbose=True,
            llm=self.llm,
            tools=self.create_tools(),
        )

    @agent
    #@handle_api_errors
    def quality_assurance_expert(self) -> Agent:
        return Agent(
            config=self.get_agent_config('quality_assurance_expert'),
            verbose=True,
            llm=self.llm,
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
        return Task(
            config=self.get_task_config('sub_processes_research_task'),
            llm=self.llm,
            agent=self.sub_processes_researcher(),  # Add parentheses to call the method
            tools=self.create_tools(),
            expected_output="List of all sub-processes",
            output_file='search_results/sub_processes.md',
        )
    
    @task
    #@handle_api_errors
    def global_regulations_research_task(self) -> Task:
        return Task(
            config=self.get_task_config('global_regulations_research_task'),
            llm=self.llm,
            agent=self.global_regulations_researcher(),
            tools=self.create_tools(),
            output_file='search_results/global_regulations.md',
            #expected_output=f"List of all global regulations under {self.topic}",
        )

    @task
    #@handle_api_errors
    def uae_regulations_research_task(self) -> Task:
        return Task(
            config=self.get_task_config('uae_regulations_research_task'),
            llm=self.llm,
            agent=self.uae_regulations_researcher(),
            tools=self.create_tools(),
            output_file='search_results/uae_regulations.md',
            #expected_output=f"List of all UAE regulations under {self.topic}",
        )

    @task
    #@handle_api_errors
    def standards_research_task(self) -> Task:
        return Task(
            config=self.get_task_config('standards_research_task'),
            llm=self.llm,
            agent=self.standards_researcher(),
            tools=self.create_tools(),
             output_file='search_results/standards.md',
            #expected_output="List of all standards",
        )

    @task
    #@handle_api_errors
    def risk_research_task(self) -> Task:
          return Task(
            config=self.get_task_config('risk_research_task'),
            llm=self.llm,
            agent=self.risk_researcher(),
            tools=self.create_tools(),
            output_file='search_results/risk_analysis.md',
            #expected_output="Detailed risk analysis with direct quotes from scraped content, source URLs, and reliability scores",
            #tools=self.create_tools()
          )
    
    @task
    #@handle_api_errors
    def quality_assurance_task(self) -> Task:
        return Task(
            config=self.get_task_config('quality_assurance_task'),
            llm=self.llm,
            agent=self.quality_assurance_expert(),
            output_file='search_results/qa_verification.md',
            #expected_output="QA verified output according to the reviewed tasks",
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.get_task_config('reporting_task'),
            llm=self.llm,
            agent=self.reporting_analyst(),
            tools=self.create_tools(),
            context=[  # Add context from all tasks
                self.sub_processes_research_task(),
                self.global_regulations_research_task(),
                self.uae_regulations_research_task(),
                self.standards_research_task(),
                self.risk_research_task(),
                self.quality_assurance_task()
            ],
            output_file='Pre_Audit_Report.md',
            async_execution=False,
            description="Compile all findings from search_results folder into report. Preserve original content verbatim.",
            callback=lambda output: logging.info(f"Report generated at: {output}"
            )
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
        """Process inputs and run the crew"""
        self.process_inputs(inputs)
        crew = self.crew()
        return crew.kickoff()