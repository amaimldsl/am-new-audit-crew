from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

@CrewBase
class PreAuditCrew():
    """PreAuditCrew crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    #llm = LLM(model="ollama/mistral")

    #crew_llm = LLM(model="ollama/mistral")

    llm = LLM(model="ollama/mistral")


    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    @agent
    def sub_processes_researcher(self) -> Agent:
        sub_processes_researcher_config = self.agents_config['sub_processes_researcher']
        
        # Access the Serper API Key from the environment
        serper_api_key = os.getenv("SERPER_API_KEY")

        # Debugging to check the API key
        #print(f"Using SERPER_API_KEY: {serper_api_key}")

        return Agent(
            role=sub_processes_researcher_config['role'],
            goal=sub_processes_researcher_config['goal'],
            backstory=sub_processes_researcher_config['backstory'],
            verbose=True,
            llm=self.llm,
            tools=[SerperDevTool(api_key=serper_api_key)],  # Pass the API key here
        )

    @agent
    def global_regulations_researcher(self) -> Agent:
        global_regulations_researcher_config = self.agents_config['global_regulations_researcher']
        
        # Access the Serper API Key from the environment
        serper_api_key = os.getenv("SERPER_API_KEY")

         # Debugging to check the API key
        #print(f"Using SERPER_API_KEY: {serper_api_key}")


        return Agent(
            role=global_regulations_researcher_config['role'],
            goal=global_regulations_researcher_config['goal'],
            backstory=global_regulations_researcher_config['backstory'],
            verbose=True,
            llm=self.llm,
            tools=[SerperDevTool(api_key=serper_api_key)],  # Pass the API key here
        )

    @agent
    def uae_regulations_researcher(self) -> Agent:
        uae_regulations_researcher_config = self.agents_config['uae_regulations_researcher']
        
        # Access the Serper API Key from the environment
        serper_api_key = os.getenv("SERPER_API_KEY")

         # Debugging to check the API key
        #print(f"Using SERPER_API_KEY: {serper_api_key}")


        return Agent(
            role=uae_regulations_researcher_config['role'],
            goal=uae_regulations_researcher_config['goal'],
            backstory=uae_regulations_researcher_config['backstory'],
            verbose=True,
            llm=self.llm,
            tools=[SerperDevTool(api_key=serper_api_key)],  # Pass the API key here
        )

    @agent
    def standards_researcher(self) -> Agent:
        standards_researcher_config = self.agents_config['standards_researcher']
        
        # Access the Serper API Key from the environment
        serper_api_key = os.getenv("SERPER_API_KEY")

         # Debugging to check the API key
        #print(f"Using SERPER_API_KEY: {serper_api_key}")


        return Agent(
            role=standards_researcher_config['role'],
            goal=standards_researcher_config['goal'],
            backstory=standards_researcher_config['backstory'],
            verbose=True,
            llm=self.llm,
            tools=[SerperDevTool(api_key=serper_api_key)],  # Pass the API key here
        )

    # Other agents remain the same as they don't need SerperDevTool

    @agent
    def quality_assurance_expert(self) -> Agent:
        quality_assurance_expert_config = self.agents_config['quality_assurance_expert']
        return Agent(
            role=quality_assurance_expert_config['role'],
            goal=quality_assurance_expert_config['goal'],
            backstory=quality_assurance_expert_config['backstory'],
            verbose=True,
            llm=self.llm,
        )

    @agent
    def reporting_analyst(self) -> Agent:
        reporting_analyst_config = self.agents_config['reporting_analyst']
        return Agent(
            role=reporting_analyst_config['role'],
            goal=reporting_analyst_config['goal'],
            backstory=reporting_analyst_config['backstory'],
            verbose=True,
            llm=self.llm,
        )

    # To learn more about structured task outputs, 
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    
    @task
    def sub_processes_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['sub_processes_research_task'],
            llm=self.llm,
          )

    @task
    def global_regulations_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['global_regulations_research_task'],
            llm=self.llm,
            
        )

    @task
    def uae_regulations_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['uae_regulations_research_task'],
            llm=self.llm,
            
        )

    @task
    def standards_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['standards_research_task'],
            llm=self.llm,
            
        )

    @task
    def quality_assurace_task(self) -> Task:
        return Task(
            config=self.tasks_config['quality_assurance_task'],
            llm=self.llm,
            
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            output_file='report.md',
            llm=self.llm,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the PreAuditCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            manager_llm=self.llm,
        )
