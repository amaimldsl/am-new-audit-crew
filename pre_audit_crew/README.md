# Pre-Audit Crew System

## Overview

The Pre-Audit Crew System is an AI-powered tool designed to generate comprehensive audit preparation materials by researching and analyzing various aspects of a specified audit subject. The system leverages multiple specialized agents to collect, process, and synthesize information from authoritative sources on the web, creating a detailed, risk-prioritized report to guide audit planning.

## Key Features

- **Multi-Agent Architecture**: Utilizes specialized AI agents for different research domains
- **Risk-Based Focus**: Prioritizes high-risk areas for efficient audit planning
- **Enhanced Language Detection**: Automatically identifies and filters non-English content
- **Website Tracking**: Maintains a database of problematic websites to avoid during research
- **Comprehensive Reporting**: Produces structured documentation with PRCT matrices

## Core Components

1. **Specialized Research Agents**:
   - Global Regulations Researcher
   - UAE Regulations Researcher
   - Sub-Processes Researcher
   - Standards Researcher
   - Risk Researcher
   - PRCT Matrix Compilation Agent
   - Reporting Analyst

2. **Enhanced Web Scraping**:
   - Automatic language detection
   - Content processing and cleanup
   - Error tracking and recovery
   - Document type filtering

3. **LLM Error Handling**:
   - Exponential backoff with jitter
   - Comprehensive error tracking
   - Performance statistics collection

## Output Files

The system generates multiple output files:
- `Pre_Audit_Report.md`: The main consolidated report with key findings
- `search_results/sub_processes.md`: Critical sub-processes analysis
- `search_results/global_regulations.md`: Global regulatory requirements
- `search_results/uae_regulations.md`: UAE-specific regulations
- `search_results/standards.md`: Industry standards analysis
- `search_results/risk_analysis.md`: Critical risk assessment
- `search_results/PRCT.md`: Consolidated Process-Risk-Control-Test matrix

## Usage

1. Run the main script:
   ```
   python src/pre_audit_crew/main.py
   ```

2. Enter the audit subject when prompted (e.g., "Basel III Compliance", "IFRS 9 Implementation")

3. The system will begin research, displaying progress updates

4. Upon completion, review the generated report and supporting files

## Configuration

- `config/agents.yaml`: Agent roles, goals, skills, and backstories
- `config/tasks.yaml`: Task descriptions, output formats, and dependencies
- `.env`: Environment variables for API keys and model settings

## Dependencies

- CrewAI: Framework for multi-agent systems
- LiteLLM: Interface for large language models
- BeautifulSoup: HTML parsing and content extraction
- Chardet: Character encoding detection
- LangDetect: Language identification (optional)

## Error Handling

The system includes robust error management:
- Language detection fallback mechanisms
- Automatic tracking of problematic websites
- Session statistics for monitoring performance
- Detailed logging for troubleshooting

This system provides a comprehensive, risk-focused pre-audit analysis that helps auditors identify critical areas requiring attention, streamlining the audit planning process and improving audit effectiveness.