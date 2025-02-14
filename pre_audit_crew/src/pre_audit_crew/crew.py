from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
import os
import time
import logging
from typing import Dict, Any
from datetime import datetime

from openai import APITimeoutError
from litellm.exceptions import Timeout
import litellm


#litellm._turn_on_debug()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pre_audit_crew.log')
    ]
)

# Error handling decorator
def handle_api_errors(func):
    def wrapper(*args, **kwargs):
        max_retries = 999
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (APIError, RateLimitError, APITimeoutError, Timeout, json.JSONDecodeError) as e:
                error_type = type(e).__name__
                if isinstance(e, json.JSONDecodeError):
                    logging.error(f"JSON parsing failed: {str(e)}")
                    time.sleep(10)
                elif isinstance(e, (APITimeoutError, Timeout)):
                    logging.warning(f"Timeout error (attempt {attempt+1}): {str(e)}")
                    time.sleep(30)
                elif isinstance(e, RateLimitError):
                    logging.warning(f"Rate limit exceeded (attempt {attempt+1}): {str(e)}")
                    time.sleep(60)
                else:
                    logging.error(f"API Error: {str(e)}")
                    raise
                
                if attempt == max_retries-1:
                    logging.error("Max retries reached. Aborting.")
                    raise
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                raise
        return func(*args, **kwargs)
    return wrapper

class DeepseekLLM(LLM):
    def __init__(self, api_key, base_url, model):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.3,
            max_retries=3,
            timeout=60,
            max_tokens=4000,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        )
    def _parse_response(self, response):
        try:
            return response.json()
        except json.JSONDecodeError:
            logging.error("Failed to parse JSON response")
            return {"error": "Invalid JSON response"}



@CrewBase
class PreAuditCrew():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, serper_api_key:str, topic: str, deepseek_api_key: str, deepseek_url: str, deepseek_model: str ):
        
        litellm.drop_params = True
        litellm.set_verbose = True
        litellm.success_callback = ["langfuse"]
            
        
        #setting needed LLMs
        
        LLAMA31_LLM = LLM(model="ollama/llama3.1")
        MISTRAL_LLM = LLM(model="ollama/mistral")
        LLAMA32_LLM = LLM(model="ollama/llama3.2")
        LLAMA33_LLM = LLM(model="ollama/llama3.3")
        MIXTRAL_LLM = LLM(model="ollama/mixtral")
        
        DEEPSEEK_LC_DF_LLM =  LLM(model="ollama/deepseek-r1")
        DEEPSEEK_LCL_14B_LLM =  LLM(model="ollama/deepseek-r1:14b")
        DEEPSEEK_LCL_32B_LLM =  LLM(model="ollama/deepseek-r1:32b")

        DEEPSEEK_LLM = DeepseekLLM(api_key=deepseek_api_key, base_url=deepseek_url, model=deepseek_model)



            
        # Configure Deepseek specifically
        #litellm.register_model(
        #    model_name="deepseek-custom",
        #    custom_llm_provider="deepseek",
        #    api_base=deepseek_url,
        #    api_key=deepseek_api_key,
        #    max_retries=3,
        #    timeout=60
        #)
        
        #DEEPSEEK_LLM = DeepseekLLM(
        #    api_key=deepseek_api_key,
        #    base_url=deepseek_url,
        #    model=deepseek_model
        #)
        


        self.llm = DEEPSEEK_LLM
        self.crew_llm = DEEPSEEK_LLM
        
        
        
        
        self.topic = topic
        self.serper_api_key = serper_api_key
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def process_inputs(self, inputs: Dict[str, Any]) -> None:
        """Process inputs received from main.py"""
        self.topic = inputs.get('topic')
        self.serper_api_key = inputs.get('serper_api_key')
        if not self.topic or not self.serper_api_key:
            raise ValueError("Missing required inputs: topic and serper_api_key must be provided")

    @handle_api_errors
    def create_tools(self) -> list:
        """Create tools with proper configuration"""
        return [
            SerperDevTool(api_key=self.serper_api_key),
            ScrapeWebsiteTool()
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

    # [Previous agent methods remain the same]
    @agent
    @handle_api_errors
    def risk_researcher(self) -> Agent:
        return Agent(
            config=self.get_agent_config('risk_researcher'),
            verbose=True,
            llm=self.llm,
            tools=self.create_tools(),
            memory=True,
            max_retries=3,
            retry_delay=60
        )

    @agent
    @handle_api_errors
    def sub_processes_researcher(self) -> Agent:
        return Agent(
            config=self.get_agent_config('sub_processes_researcher'),
            verbose=True,
            llm=self.llm,
            tools=self.create_tools(),
        )

    @agent
    @handle_api_errors
    def global_regulations_researcher(self) -> Agent:
        return Agent(
            config=self.get_agent_config('global_regulations_researcher'),
            verbose=True,
            llm=self.llm,
            tools=self.create_tools(),
        )

    @agent
    @handle_api_errors
    def uae_regulations_researcher(self) -> Agent:
        return Agent(
            config=self.get_agent_config('uae_regulations_researcher'),
            verbose=True,
            llm=self.llm,
            tools=self.create_tools(),
        )

    @agent
    @handle_api_errors
    def standards_researcher(self) -> Agent:
        return Agent(
            config=self.get_agent_config('standards_researcher'),
            verbose=True,
            llm=self.llm,
            tools=self.create_tools(),
        )

    @agent
    @handle_api_errors
    def quality_assurance_expert(self) -> Agent:
        return Agent(
            config=self.get_agent_config('quality_assurance_expert'),
            verbose=True,
            llm=self.llm,
        )

    @agent
    @handle_api_errors
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.get_agent_config('reporting_analyst'),
            verbose=True,
            llm=self.llm,
        )


    @task
    @handle_api_errors
    def sub_processes_research_task(self) -> Task:
        return Task(
            config=self.get_task_config('sub_processes_research_task'),
            llm=self.llm,
            expected_output="List of all sub-processes",
        )

    @task
    @handle_api_errors
    def global_regulations_research_task(self) -> Task:
        return Task(
            config=self.get_task_config('global_regulations_research_task'),
            llm=self.llm,
            expected_output=f"List of all global regulations under {self.topic}",
        )

    @task
    @handle_api_errors
    def uae_regulations_research_task(self) -> Task:
        return Task(
            config=self.get_task_config('uae_regulations_research_task'),
            llm=self.llm,
            expected_output=f"List of all UAE regulations under {self.topic}",
        )

    @task
    @handle_api_errors
    def standards_research_task(self) -> Task:
        return Task(
            config=self.get_task_config('standards_research_task'),
            llm=self.llm,
            expected_output="List of all standards",
        )

    @task
    @handle_api_errors
    def risk_research_task(self) -> Task:
        return Task(
            config=self.get_task_config('risk_research_task'),
            llm=self.llm,
            expected_output="Top 10 risks with likelihood, impact, and source URLs",
            tools=self.create_tools(),
        )
    
    @task
    @handle_api_errors
    def quality_assurance_task(self) -> Task:
        return Task(
            config=self.get_task_config('quality_assurance_task'),
            llm=self.llm,
            expected_output="QA verified output according to the reviewed tasks",
        )

    @task
    @handle_api_errors
    def reporting_task(self) -> Task:
        return Task(
            config=self.get_task_config('reporting_task'),
            llm=self.llm,
            output_file='report.md',
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