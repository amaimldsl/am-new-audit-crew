from crewai import Agent, Crew, Process, Task,LLM
from crewai.project import CrewBase, agent, crew, task

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class PreAuditCrew():
	"""PreAuditCrew crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'
	
	llm = LLM(model="ollama/mistral")

	crew_llm = LLM(model="ollama/phi4")

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def global_regulations_researcher(self) -> Agent:
		global_regulations_researcher_config = self.agents_config['global_regulations_researcher']
		return Agent(
			role=global_regulations_researcher_config['role'],
			goal=global_regulations_researcher_config['goal'],
			backstory=global_regulations_researcher_config['backstory'],
			verbose=True,
			llm=self.llm,
		)
	

	@agent
	def uae_regulations_researcher(self) -> Agent:
		uae_regulations_researcher_config = self.agents_config['uae_regulations_researcher']
		return Agent(
			role=uae_regulations_researcher_config['role'],
			goal=uae_regulations_researcher_config['goal'],
			backstory=uae_regulations_researcher_config['backstory'],
			verbose=True,
			llm=self.llm,
		)
	

	@agent
	def sub_processes_researcher(self) -> Agent:
		sub_processes_researcher_config = self.agents_config['sub_processes_researcher']
		return Agent(
			role=sub_processes_researcher_config['role'],
			goal=sub_processes_researcher_config['goal'],
			backstory=sub_processes_researcher_config['backstory'],
			verbose=True,
			llm=self.llm,
		)
	
	@agent
	def standards_researcher(self) -> Agent:
		standards_researcher_config = self.agents_config['standards_researcher']
		return Agent(
			role=standards_researcher_config['role'],
			goal=standards_researcher_config['goal'],
			backstory=standards_researcher_config['backstory'],
			verbose=True,
			llm=self.llm,
		)

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
	def global_regulations_research_task(self) -> Task:
		return Task(
			config=self.tasks_config['global_regulations_research_task'],
			llm=self.llm,
			expected_output="list of all global regulations under {topic} ",
		)


	@task
	def uae_regulations_research_task(self) -> Task:
		return Task(
			config=self.tasks_config['uae_regulations_research_task'],
			llm=self.llm,
			expected_output="list of all UAE regulations under {topic} ",
		)

	@task
	def sub_processes_research_task(self) -> Task:
		return Task(
			config=self.tasks_config['sub_processes_research_task'],
			llm=self.llm,
			expected_output="list of all subprocesses"
		)

	@task
	def standards_research_task(self) -> Task:
		return Task(
			config=self.tasks_config['standards_research_task'],
			llm=self.llm,
			expected_output="list of all standards",
		)



	@task
	def quality_assurace_task(self) -> Task:
		return Task(
			config=self.tasks_config['quality_assurance_task'],
			llm=self.llm,
			expected_output=" QA verified output according to the reviewed tasks",
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
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			#process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
			#manager_agent=self.quality_assurace_task,
			manager_llm=self.crew_llm,
		)
