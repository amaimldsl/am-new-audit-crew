from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, FileReadTool, DirectoryReadTool
import os
import time
import logging
from typing import Dict, Any
from datetime import datetime
from litellm.exceptions import RateLimitError, APIError
import litellm
import json
import re
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse

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
        self.skipped_urls = set()  # Track URLs causing errors

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

class EnhancedSerperDevTool(SerperDevTool):
    """Enhanced Serper Dev Tool with focus on quality sources and recency."""
    
    def __init__(self, api_key, premium_sources=None):
        super().__init__(api_key=api_key)
        self.premium_sources = premium_sources or [
            "gartner.com", "accenture.com", "mckinsey.com", "deloitte.com", 
            "pwc.com", "ey.com", "kpmg.com", "bis.org", "imf.org", "worldbank.org",
            "gov", "org", "edu", "iso.org", "iec.ch", "nist.gov"
        ]
        
    def _run(self, query, **kwargs):
        """Enhanced search with quality filters and recency focus."""
        # Add recency parameter to query
        if 'after:' not in query and 'before:' not in query:
            # Default to content from the last 2 years
            current_year = datetime.now().year
            query = f"{query} after:{current_year-2}"
        
        # Add preference for reliable sources
        source_boost = " OR ".join([f"site:{source}" for source in self.premium_sources[:5]])
        query = f"{query} ({source_boost})"
        
        # Get search results
        results = super()._run(query, **kwargs)
        
        # Process results to improve quality
        try:
            results_dict = json.loads(results)
            
            # Score and rank results
            if 'organic' in results_dict:
                for result in results_dict['organic']:
                    result['quality_score'] = self._calculate_quality_score(result)
                
                # Sort by quality score
                results_dict['organic'] = sorted(
                    results_dict['organic'], 
                    key=lambda x: x.get('quality_score', 0), 
                    reverse=True
                )
                
                # Take top 10 results
                results_dict['organic'] = results_dict['organic'][:10]
                
            # Convert back to string
            return json.dumps(results_dict)
        except Exception as e:
            logging.error(f"Error processing search results: {str(e)}")
            return results
    
    def _calculate_quality_score(self, result):
        """Calculate quality score for a search result."""
        score = 0
        
        # Domain authority score
        domain = urlparse(result.get('link', '')).netloc
        if any(trusted in domain for trusted in self.premium_sources):
            score += 30
        
        # Title relevance
        if 'risk' in result.get('title', '').lower():
            score += 15
        if 'compliance' in result.get('title', '').lower():
            score += 10
        if 'audit' in result.get('title', '').lower():
            score += 10
            
        # Recency score (if available)
        if 'date' in result:
            try:
                date_str = result['date']
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                days_old = (datetime.now() - date_obj).days
                recency_score = max(0, 20 - (days_old / 30))  # Up to 20 points for recent content
                score += recency_score
            except:
                pass
        
        return score

class EnhancedScraper(ScrapeWebsiteTool):
    """Enhanced scraper that extracts important content and handles errors gracefully."""
    
    def __init__(self, crew, **kwargs):
        super().__init__(**kwargs)
        self._crew = crew
        
    def _run(self, website_url: str, *args, **kwargs) -> str:
        """Extract high-value content from websites."""
        if website_url in self._crew.skipped_urls:
            return f"Skipped URL {website_url}"
        
        try:
            # Try to get full content first
            content = super()._run(website_url=website_url)
            
            # Check if content is too large
            if len(content) > 10000:
                # Try to extract the important parts
                content = self._extract_important_content(website_url, content)
            
            if len(content) > 3500:
                self._crew.skipped_urls.add(website_url)
                raise ContentTooLargeError(f"Content too large: {website_url}")
                
            return content
        except Exception as e:
            if "413" in str(e) or "Too Large" in str(e):
                self._crew.skipped_urls.add(website_url)
            raise
    
    def _extract_important_content(self, url, content):
        """Extract the most important parts of the page focusing on risk-related content."""
        try:
            # Try to parse with BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove scripts, styles, and other non-content elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            
            # Look for risk-related sections
            risk_keywords = ["risk", "compliance", "audit", "control", "regulation", 
                           "standard", "requirement", "hazard", "danger", "warning"]
            
            important_paragraphs = []
            
            # Look for headers with risk keywords
            for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                header_text = header.get_text().lower()
                if any(keyword in header_text for keyword in risk_keywords):
                    # Get this header and the content that follows it
                    section = [header.get_text().strip()]
                    next_element = header.find_next_sibling()
                    while next_element and next_element.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                        if next_element.name in ['p', 'li', 'div'] and len(next_element.get_text().strip()) > 20:
                            section.append(next_element.get_text().strip())
                        next_element = next_element.find_next_sibling()
                    
                    important_paragraphs.append("\n".join(section))
            
            # If we didn't find risk-related sections, take the main content
            if not important_paragraphs:
                main_content = soup.find('main') or soup.find('article') or soup.find('body')
                if main_content:
                    for p in main_content.find_all(['p', 'li']):
                        text = p.get_text().strip()
                        if len(text) > 50:  # Only take substantial paragraphs
                            important_paragraphs.append(text)
            
            # Limit to a reasonable size
            result = "\n\n".join(important_paragraphs[:10])
            
            # Add page title and URL for reference
            page_title = soup.title.string if soup.title else "Unknown Title"
            result = f"Page Title: {page_title}\nURL: {url}\n\n{result}"
            
            return result
            
        except Exception as e:
            logging.error(f"Error extracting important content: {str(e)}")
            # Fall back to simple truncation if parsing fails
            return content[:3000] + "...[content truncated]"


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
            temperature=0.2,  # Lower temperature for more factual responses
            max_tokens=4000,
            max_retries=999,  # High retry count for persistence
            base_delay=2,
            max_delay=61
        )
        self.crew_llm = self.llm  # Use same LLM instance for crew
        self.topic = topic
        self.serper_api_key = serper_api_key
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create output directories if they don't exist
        os.makedirs('./search_results', exist_ok=True)
    
    def create_tools(self) -> list:
        return [
            EnhancedSerperDevTool(api_key=self.serper_api_key),
            EnhancedScraper(
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
                results = self.sub_processes_researcher.execute(context=context)
                self._save_top_items("sub_processes", results)
                return results
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
            execution_fn=_custom_execute,
        )

    @task
    def global_regulations_research_task(self) -> Task:
        def _custom_execute(context=None):
            try:
                results = self.global_regulations_researcher.execute(context=context)
                self._save_top_items("global_regulations", results)
                return results
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
            execution_fn=_custom_execute,
        )

    @task
    def uae_regulations_research_task(self) -> Task:
        def _custom_execute(context=None):
            try:
                results = self.uae_regulations_researcher.execute(context=context)
                self._save_top_items("uae_regulations", results)
                return results
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
            execution_fn=_custom_execute,
        )

    @task
    def standards_research_task(self) -> Task:
        def _custom_execute(context=None):
            try:
                results = self.standards_researcher.execute(context=context)
                self._save_top_items("standards", results)
                return results
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
            execution_fn=_custom_execute,
        )

    @task
    def risk_research_task(self) -> Task:
        def _custom_execute(context=None):
            try:
                results = self.risk_researcher.execute(context=context)
                self._save_top_items("risk_analysis", results)
                return results
            except ContentTooLargeError as e:
                current_url = str(e).split("Skipped URL: ")[-1]
                self.skipped_urls.add(current_url)
                logging.warning(f"Permanently skipped: {current_url}")
                return f"Skipped website: {current_url}"

        return Task(
            config=self.get_task_config('risk_research_task'),
            llm=self.llm,
            agent=self.risk_researcher(),
            tools=self.create_tools(),
            execution_fn=_custom_execute,
        )
          
    def _save_top_items(self, category, content):
        """Process content to extract and save top 10 high-risk items."""
        try:
            # Extract PRCT matrix entries
            prct_entries = self._extract_prct_entries(content)
            
            # Sort entries by risk rating (High -> Medium -> Low)
            risk_priority = {"High": 3, "Medium": 2, "Low": 1}
            sorted_entries = sorted(
                prct_entries, 
                key=lambda x: risk_priority.get(x.get("Risk Rating", ""), 0),
                reverse=True
            )
            
            # Take top 10 entries
            top_entries = sorted_entries[:10]
            
            # Format for saving
            formatted_content = f"# Top 10 {category.replace('_', ' ').title()} Items\n\n"
            
            for i, entry in enumerate(top_entries, 1):
                formatted_content += f"## {i}. {entry.get('Process', 'Unnamed Process')}\n\n"
                
                if "Source" in entry:
                    formatted_content += f"**Source:** {entry['Source']}\n\n"
                
                for key in ["Risk", "Risk Rating", "Control", "Test"]:
                    if key in entry:
                        formatted_content += f"**{key}:** {entry[key]}\n\n"
                
                formatted_content += "---\n\n"
            
            # Save to file
            with open(f"./search_results/top10_{category}.md", "w") as f:
                f.write(formatted_content)
                
            logging.info(f"Saved top 10 {category} items")
            
        except Exception as e:
            logging.error(f"Error saving top items for {category}: {str(e)}")
    
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
            description="Compile all findings with focus on top 10 risk points in each category",
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