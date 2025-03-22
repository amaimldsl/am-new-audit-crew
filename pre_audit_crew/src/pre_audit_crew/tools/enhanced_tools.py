import os
import pickle
import logging
import time
import re
import string
import requests
from urllib.parse import urlparse
from datetime import datetime
from typing import Set, Dict, List, Any, Union
from bs4 import BeautifulSoup
import chardet
from crewai import LLM
from crewai_tools import ScrapeWebsiteTool
from litellm.exceptions import RateLimitError, APIError
import litellm
from crewai_tools import ScrapeWebsiteTool




def create_enhanced_scrape_tool(tracker_manager=None, language_detector=None, **kwargs):
    """Factory function to create an enhanced scraping tool"""
    
    # Create a standard scraping tool with the original configuration
    original_tool = ScrapeWebsiteTool(**kwargs)
    
    # Store the original _run method
    original_run = original_tool._run
    
    # Define a new _run method that adds our enhancements
    def enhanced_run(**run_kwargs):
        try:
            # Get the website_url from kwargs
            website_url = run_kwargs.get('website_url')
            if not website_url:
                return "Missing website_url parameter"
                
            # Check if URL is in blocklist
            if tracker_manager and tracker_manager.is_blocked(website_url):
                logging.warning(f"Skipping previously blocked website: {website_url}")
                return f"⚠️ This website was previously blocked: {website_url}"
                
            # Check for document files
            doc_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.zip', '.rar']
            if any(website_url.lower().endswith(ext) for ext in doc_extensions):
                if tracker_manager:
                    tracker_manager.add_blocked_website(website_url, reason="document-file")
                logging.warning(f"Skipping document file: {website_url}")
                return f"⚠️ This URL points to a document file which cannot be processed by the web scraper: {website_url}"
                
            # Get the content using the original method
            content = original_run(**run_kwargs)
            
            # Check if content is English
            if language_detector and content and not language_detector.check_url_content(website_url, content):
                if tracker_manager:
                    tracker_manager.add_blocked_website(website_url, reason="non-english")
                logging.warning(f"Non-English content detected for {website_url}")
                return f"⚠️ This website contains primarily non-English content and has been skipped."
            
            # Process the content if it exists
            if content:
                return process_content(website_url, content)
            
            return content
            
        except Exception as e:
            if tracker_manager:
                tracker_manager.record_llm_error(
                    run_kwargs.get('website_url', 'unknown'), 
                    type(e).__name__, 
                    str(e)
                )
            logging.error(f"Error scraping {run_kwargs.get('website_url')}: {str(e)}")
            return f"⚠️ Error scraping website: {run_kwargs.get('website_url')}. Error: {str(e)}"
    
    def process_content(url, content, max_length=8000):
        """Process website content to make it more manageable"""
        try:
            # Use BeautifulSoup to extract the main content
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script, style, meta and other non-content tags
            for tag in soup(['script', 'style', 'meta', 'head', 'footer', 'nav', 'iframe', 'noscript']):
                tag.decompose()
            
            # Extract text
            text = soup.get_text(separator=' ')
            
            # Clean up whitespace
            text = ' '.join(text.split())
            
            # Truncate if still too long
            if len(text) > max_length:
                text = text[:max_length] + "... [content truncated due to length]"
            
            return text
            
        except Exception as e:
            logging.error(f"Error processing content from {url}: {e}")
            # If processing fails, just truncate
            if len(content) > max_length:
                return content[:max_length] + "... [content truncated due to processing error]"
            return content
    
    # Replace the original _run method with our enhanced version
    original_tool._run = enhanced_run
    
    return original_tool





class EnhancedWebsiteHandler:
    """Helper class for enhanced language detection and error tracking"""
    
    def __init__(self, tracker_manager=None, language_detector=None):
        self.tracker_manager = tracker_manager
        self.language_detector = language_detector

    def process_website(self, website_url, content):
        """Process website content with language detection and error tracking"""
        if not website_url:
            return "Missing website_url parameter"
                
        # Check if URL is in blocklist
        if self.tracker_manager and self.tracker_manager.is_blocked(website_url):
            logging.warning(f"Skipping previously blocked website: {website_url}")
            return f"⚠️ This website was previously blocked: {website_url}"
            
        # Check for document files
        doc_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.zip', '.rar']
        if any(website_url.lower().endswith(ext) for ext in doc_extensions):
            if self.tracker_manager:
                self.tracker_manager.add_blocked_website(website_url, reason="document-file")
            logging.warning(f"Skipping document file: {website_url}")
            return f"⚠️ This URL points to a document file which cannot be processed by the web scraper: {website_url}"
        
        # Check if content is English
        if self.language_detector and not self.language_detector.check_url_content(website_url, content):
            if self.tracker_manager:
                self.tracker_manager.add_blocked_website(website_url, reason="non-english")
            logging.warning(f"Non-English content detected for {website_url}")
            return f"⚠️ This website contains primarily non-English content and has been skipped."
        
        # Process and return the content
        return self._process_content(website_url, content)
    
    def _process_content(self, url, content, max_length=8000):
        """Pre-process website content to make it more manageable"""
        try:
            # Use BeautifulSoup to extract the main content
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script, style, meta and other non-content tags
            for tag in soup(['script', 'style', 'meta', 'head', 'footer', 'nav', 'iframe', 'noscript']):
                tag.decompose()
            
            # Extract text
            text = soup.get_text(separator=' ')
            
            # Clean up whitespace
            text = ' '.join(text.split())
            
            # Truncate if still too long
            if len(text) > max_length:
                text = text[:max_length] + "... [content truncated due to length]"
            
            return text
            
        except Exception as e:
            logging.error(f"Error processing content from {url}: {e}")
            # If processing fails, just truncate
            if len(content) > max_length:
                return content[:max_length] + "... [content truncated due to processing error]"
            return content

# Use the standard scraping tool but with a post-processor
def create_enhanced_scrape_tool(tracker_manager=None, language_detector=None, **kwargs):
    """Factory function to create an enhanced scraping tool"""
    
    # Create our handler
    handler = EnhancedWebsiteHandler(
        tracker_manager=tracker_manager,
        language_detector=language_detector
    )
    
    # Create a standard scraping tool
    standard_tool = ScrapeWebsiteTool(**kwargs)
    
    # Store the original _run method
    original_run = standard_tool._run
    
    # Override the _run method to add our enhancements
    def enhanced_run(**kwargs):
        website_url = kwargs.get('website_url')
        
        # First check if URL is in blocklist
        if tracker_manager and tracker_manager.is_blocked(website_url):
            return f"⚠️ This website was previously blocked: {website_url}"
            
        # Check for document files
        doc_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.zip', '.rar']
        if any(website_url.lower().endswith(ext) for ext in doc_extensions):
            if tracker_manager:
                tracker_manager.add_blocked_website(website_url, reason="document-file")
            return f"⚠️ This URL points to a document file which cannot be processed: {website_url}"
        
        try:
            # Call the original implementation
            content = original_run(**kwargs)
            
            # Now process with our enhancements
            if content and website_url:
                return handler.process_website(website_url, content)
            return content
        except Exception as e:
            if tracker_manager:
                tracker_manager.record_llm_error(
                    website_url, 
                    type(e).__name__, 
                    str(e)
                )
            logging.error(f"Error scraping {website_url}: {str(e)}")
            return f"⚠️ Error scraping website: {website_url}. Error: {str(e)}"
    
    # Replace the method
    standard_tool._run = enhanced_run
    
    return standard_tool




#########################
try:
    from langdetect import detect, LangDetectException
except ImportError:
    logging.warning("langdetect not installed. Using fallback character-based detection.")
    detect = None

# Custom exceptions
class ContentTooLargeError(Exception):
    """Raised when content is too large for LLM processing"""
    pass

class NonEnglishContentError(Exception):
    """Raised when content is primarily non-English"""
    pass

class EnhancedLanguageDetection:
    """Enhanced language detection that's more accurate than character-based methods"""
    
    # Keep a cache of previously checked URLs and their language
    _language_cache = {}
    
    @staticmethod
    def is_english(text, min_text_length=100, confidence_sample_size=500):
        """
        Determine if text is in English using langdetect
        
        Args:
            text (str): Text to analyze
            min_text_length (int): Minimum text length required for reliable detection
            confidence_sample_size (int): Size of text sample used for detection
        
        Returns:
            bool: True if text is in English, False otherwise
        """
        if not text or len(text) < min_text_length:
            return True  # Too short to determine, assume English
            
        try:
            # Clean the text for better detection
            clean_text = ' '.join(re.sub(r'[^\w\s]', ' ', text).split())
            
            # For very long texts, just use a sample to speed up detection
            if len(clean_text) > confidence_sample_size:
                # Take samples from the beginning, middle and end for better accuracy
                beginning = clean_text[:confidence_sample_size//3]
                middle_start = len(clean_text)//2 - confidence_sample_size//6
                middle = clean_text[middle_start:middle_start + confidence_sample_size//3]
                end = clean_text[-confidence_sample_size//3:]
                sample_text = beginning + " " + middle + " " + end
            else:
                sample_text = clean_text
            
            # Use langdetect if available
            if detect:
                # Detect language
                detected_lang = detect(sample_text)
                logging.debug(f"Detected language: {detected_lang}")
                
                # Return True if English, False otherwise
                return detected_lang == 'en'
            else:
                # Fallback to character-based method
                return EnhancedLanguageDetection._is_mainly_english_chars(text)
            
        except Exception as e:
            logging.warning(f"Language detection error: {str(e)}")
            # Fall back to character-based method if langdetect fails
            return EnhancedLanguageDetection._is_mainly_english_chars(text)
            
    @staticmethod
    def _is_mainly_english_chars(text, threshold=0.7):
        """Fallback method using character distribution to estimate if text is English"""
        if not text or len(text) < 100:
            return True
            
        # Count ASCII characters (roughly English)
        ascii_count = sum(1 for c in text if c in string.printable)
        return ascii_count / len(text) >= threshold
    
    @classmethod
    def check_url_content(cls, url, content):
        """
        Check if content from URL is in English.
        Caches results for performance.
        
        Args:
            url (str): Source URL
            content (str): Text content to analyze
            
        Returns:
            bool: True if content is in English, False otherwise
        """
        # Check cache first
        if url in cls._language_cache:
            return cls._language_cache[url]
            
        # Clean content with BeautifulSoup
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script, style, meta and other non-content tags
            for tag in soup(['script', 'style', 'meta', 'head', 'footer', 'nav']):
                tag.decompose()
                
            # Get text content
            text = soup.get_text()
            text = ' '.join(text.split())
            
            # Check if content is English
            is_english = cls.is_english(text)
            
            # Cache result
            cls._language_cache[url] = is_english
            
            return is_english
            
        except Exception as e:
            logging.error(f"Error processing content from {url}: {e}")
            # Default to True to avoid unnecessarily blocking websites
            return True

class WebsiteTrackingManager:
    """Enhanced manager to track problematic websites and LLM errors"""
    
    # Document extensions that should be blocked by default
    BLOCKED_EXTENSIONS = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.zip', '.rar']
    
    def __init__(self, blocklist_file='blocked_websites.pkl', error_log_file='llm_errors.pkl'):
        self.blocklist_file = blocklist_file
        self.error_log_file = error_log_file
        self.blocklist = self._load_file(blocklist_file, set())
        self.error_log = self._load_file(error_log_file, {})
        
        # Track session-specific errors
        self.session_errors = set()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _load_file(self, filename: str, default_value: Any) -> Any:
        """Load data from disk if it exists"""
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    return pickle.load(f)
            return default_value
        except Exception as e:
            logging.error(f"Error loading file {filename}: {str(e)}")
            return default_value
    
    def _save_file(self, data: Any, filename: str) -> None:
        """Save data to disk"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            logging.info(f"Saved data to {filename}")
        except Exception as e:
            logging.error(f"Error saving to {filename}: {str(e)}")
    
    def save_blocklist(self) -> None:
        """Save the blocklist to disk"""
        self._save_file(self.blocklist, self.blocklist_file)
        
    def save_error_log(self) -> None:
        """Save the error log to disk"""
        self._save_file(self.error_log, self.error_log_file)
        
    def add_blocked_website(self, url: str, reason: str = "unknown") -> None:
        """
        Add a website to the blocklist with reason
        
        Args:
            url (str): Website URL to block
            reason (str): Reason for blocking (e.g., "non-english", "llm-error", "size-limit")
        """
        if not url or url == 'unknown':
            return
            
        parsed = urlparse(url)
        normalized_url = f"{parsed.netloc}{parsed.path}"
        
        # Add to blocklist with timestamp and reason
        self.blocklist.add(normalized_url)
        
        # Track in error log with details
        if normalized_url not in self.error_log:
            self.error_log[normalized_url] = []
            
        self.error_log[normalized_url].append({
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'session_id': self.session_id
        })
        
        # Add to session errors
        self.session_errors.add(normalized_url)
        
        # Save changes
        self.save_blocklist()
        self.save_error_log()
        
        logging.info(f"Added {normalized_url} to blocklist. Reason: {reason}")
    
    def record_llm_error(self, url: str, error_type: str, error_message: str) -> None:
        """
        Record an LLM error for a specific URL
        
        Args:
            url (str): The URL that caused the error
            error_type (str): Type of error (e.g., "RateLimitError", "APIError")
            error_message (str): The error message
        """
        if not url or url == 'unknown':
            url = 'unknown-url'
            
        parsed = urlparse(url)
        normalized_url = f"{parsed.netloc}{parsed.path}"
        
        # Add to error log
        if normalized_url not in self.error_log:
            self.error_log[normalized_url] = []
            
        self.error_log[normalized_url].append({
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_message[:200],  # Truncate long messages
            'session_id': self.session_id
        })
        
        # Save error log
        self.save_error_log()
        
        # If the same URL has caused multiple errors, consider blocking it
        error_count = sum(1 for entry in self.error_log[normalized_url] 
                         if entry.get('session_id') == self.session_id)
                         
        if error_count >= 3:  # Block after 3 errors in same session
            self.add_blocked_website(url, reason=f"repeated-errors-{error_type}")
            logging.warning(f"Auto-blocked {normalized_url} after {error_count} errors in this session")
    
    def is_blocked(self, url: str) -> bool:
        """Check if a website is in the blocklist or has a blocked extension"""
        if not url or url == 'unknown':
            return False
        
        # Check for document extensions that should be blocked
        if any(url.lower().endswith(ext) for ext in self.BLOCKED_EXTENSIONS):
            logging.info(f"Blocked document file: {url}")
            return True
            
        parsed = urlparse(url)
        normalized_url = f"{parsed.netloc}{parsed.path}"
        return normalized_url in self.blocklist
    
    def get_error_summary(self) -> Dict:
        """
        Get summary of errors
        
        Returns:
            Dict: Summary of errors
        """
        return {
            'total_blocked_sites': len(self.blocklist),
            'total_error_sites': len(self.error_log),
            'total_errors': sum(len(errors) for errors in self.error_log.values()),
            'session_errors': len(self.session_errors)
        }
    
    def get_session_stats(self) -> Dict:
        """Get statistics for the current session"""
        session_error_count = 0
        session_blocked_count = 0
        
        # Count errors and blocks from this session
        for url, errors in self.error_log.items():
            session_errors = [e for e in errors if e.get('session_id') == self.session_id]
            session_error_count += len(session_errors)
            
            # Count blocks from this session
            if url in self.blocklist and url in self.session_errors:
                session_blocked_count += 1
                
        return {
            'session_id': self.session_id,
            'error_count': session_error_count,
            'blocked_count': session_blocked_count,
            'total_blocked_sites': len(self.blocklist)
        }
    
    def get_blocklist_summary(self) -> str:
        """Get a summary of the blocklist"""
        return f"{len(self.blocklist)} websites are blocked due to previous errors"

class EnhancedScrapeWebsiteTool(ScrapeWebsiteTool):
    """Enhanced web scraper with improved language detection and error handling"""
    
    def __init__(self, tracker_manager=None, language_detector=None, **kwargs):
        # Extract managers before calling super().__init__
        self._tracker_manager = tracker_manager
        self._language_detector = language_detector
        
        # Default configuration with reasonable values
        default_config = {
            "max_retries": 2,
            "suppress_errors": True,
            "ssl_verify": False,
            "timeout": 15,  # 15 second timeout
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
        }
        
        # Merge default config with provided config
        if "config" in kwargs:
            for key, value in default_config.items():
                if key not in kwargs["config"]:
                    kwargs["config"][key] = value
        else:
            kwargs["config"] = default_config
            
        super().__init__(**kwargs)
    
    def _run(self, **kwargs):
        """Enhanced run method with better error handling and language detection"""
        try:
            # Get the website_url from kwargs
            website_url = kwargs.get('website_url')
            if not website_url:
                raise ValueError("Missing website_url parameter")
                
            # Check if URL is in blocklist
            if self._tracker_manager and self._tracker_manager.is_blocked(website_url):
                logging.warning(f"Skipping previously blocked website: {website_url}")
                return f"⚠️ This website was previously blocked: {website_url}"
                
            # Check for document files
            doc_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.zip', '.rar']
            if any(website_url.lower().endswith(ext) for ext in doc_extensions):
                if self._tracker_manager:
                    self._tracker_manager.add_blocked_website(website_url, reason="document-file")
                logging.warning(f"Skipping document file: {website_url}")
                return f"⚠️ This URL points to a document file which cannot be processed by the web scraper: {website_url}"
                
            # Enhanced request with proper headers and error handling
            try:
                headers = {
                    'User-Agent': self.config.get('user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
                
                response = requests.get(
                    website_url, 
                    headers=headers,
                    verify=self.config.get('ssl_verify', False),
                    timeout=self.config.get('timeout', 15)
                )
                
                if response.status_code != 200:
                    if self._tracker_manager:
                        self._tracker_manager.record_llm_error(
                            website_url, 
                            "HTTPError", 
                            f"Status code: {response.status_code}"
                        )
                    return f"⚠️ Failed to retrieve website. Status code: {response.status_code}"
                
                # Detect encoding if not specified
                if response.encoding is None or response.encoding == 'ISO-8859-1':
                    detected_encoding = chardet.detect(response.content)
                    response.encoding = detected_encoding['encoding']
                
                content = response.text
                
                # Check if content is English
                if self._language_detector and not self._language_detector.check_url_content(website_url, content):
                    if self._tracker_manager:
                        self._tracker_manager.add_blocked_website(website_url, reason="non-english")
                    logging.warning(f"Non-English content detected for {website_url}")
                    return f"⚠️ This website contains primarily non-English content and has been skipped."
                
                # Process the content
                content = self._process_content(website_url, content)
                return content
                
            except requests.RequestException as e:
                if self._tracker_manager:
                    self._tracker_manager.record_llm_error(website_url, "RequestError", str(e))
                logging.error(f"Request error for {website_url}: {str(e)}")
                return f"⚠️ Error retrieving website: {str(e)}"
            
        except Exception as e:
            if self._tracker_manager:
                self._tracker_manager.record_llm_error(
                    kwargs.get('website_url', 'unknown'), 
                    type(e).__name__, 
                    str(e)
                )
            logging.error(f"Error scraping {kwargs.get('website_url')}: {str(e)}")
            return f"⚠️ Error scraping website: {kwargs.get('website_url')}. Error: {str(e)}"
    
    def _process_content(self, url, content, max_length=8000):
        """Pre-process website content to make it more manageable"""
        try:
            # Use BeautifulSoup to extract the main content
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script, style, meta and other non-content tags
            for tag in soup(['script', 'style', 'meta', 'head', 'footer', 'nav', 'iframe', 'noscript']):
                tag.decompose()
            
            # Extract text
            text = soup.get_text(separator=' ')
            
            # Clean up whitespace
            text = ' '.join(text.split())
            
            # Truncate if still too long
            if len(text) > max_length:
                text = text[:max_length] + "... [content truncated due to length]"
            
            return text
            
        except Exception as e:
            logging.error(f"Error processing content from {url}: {e}")
            # If processing fails, just truncate
            if len(content) > max_length:
                return content[:max_length] + "... [content truncated due to processing error]"
            return content

class EnhancedLLM(LLM):
    """Enhanced LLM with better error handling and tracking"""
    
    def __init__(self, *args, tracker_manager=None, max_retries=5, base_delay=2, max_delay=61, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker_manager = tracker_manager
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
        # Initialize tracking metrics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.retry_counts = []
        self.error_types = {}

    def call(self, *args, **kwargs):
        """
        Enhanced call method with better error handling and tracking
        
        Args:
            *args: Arguments to pass to the LLM
            **kwargs: Keyword arguments to pass to the LLM
            
        Returns:
            The LLM response
            
        Raises:
            Various exceptions that can occur during LLM calls
        """
        retry_count = 0
        self.total_calls += 1
        current_url = kwargs.get('context', {}).get('website_url', 'unknown')
        start_time = time.time()
        
        while retry_count < self.max_retries:
            try:
                # Attempt the LLM call
                response = super().call(*args, **kwargs)
                
                # Track successful call
                self.successful_calls += 1
                self.retry_counts.append(retry_count)
                
                # Calculate and log duration
                duration = time.time() - start_time
                logging.info(f"LLM call successful after {retry_count} retries. Duration: {duration:.2f}s")
                
                return response
                
            except (RateLimitError, litellm.exceptions.RateLimitError) as e:
                # Handle rate limit errors
                retry_count += 1
                error_type = "RateLimitError"
                error_message = str(e)
                
                # Track error type
                self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
                
                # Log and record the error
                logging.warning(f"Rate limit error on attempt {retry_count}/{self.max_retries}: {error_message[:100]}...")
                
                if self.tracker_manager:
                    self.tracker_manager.record_llm_error(current_url, error_type, error_message)
                
                if retry_count >= self.max_retries:
                    self.failed_calls += 1
                    logging.error("Max retries reached for rate limit, raising error")
                    raise
                    
                # Exponential backoff with jitter
                delay = min(self.base_delay * (2 ** retry_count), self.max_delay)
                jitter = delay * 0.1 * (0.5 - time.time() % 1)  # Small random jitter
                delay_with_jitter = delay + jitter
                
                logging.info(f"Rate limited. Retrying in {delay_with_jitter:.2f} seconds...")
                time.sleep(delay_with_jitter)
                
            except (APIError, litellm.exceptions.APIError) as e:
                # Handle API errors
                retry_count += 1
                error_message = str(e)
                error_type = "APIError"
                
                # Track error type
                self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
                
                # Check if this is a content too large error
                if hasattr(e, 'status_code') and e.status_code == 413 or "413 Request Entity Too Large" in error_message:
                    logging.error(f"Content too large, skipping website. Error: {error_message[:100]}...")
                    
                    # Record the error and block the website
                    if self.tracker_manager:
                        self.tracker_manager.add_blocked_website(current_url, reason="content-too-large")
                    
                    self.failed_calls += 1
                    raise ContentTooLargeError(f"Content too large for LLM processing: {current_url}") from e
                
                # Check for invalid JSON responses
                if "Expecting value: line 1 column 1 (char 0)" in error_message:
                    logging.warning(f"JSON parsing error detected (attempt {retry_count})")
                
                # Log and record the error
                logging.error(f"API error on attempt {retry_count}/{self.max_retries}: {error_message[:100]}...")
                
                if self.tracker_manager:
                    self.tracker_manager.record_llm_error(current_url, error_type, error_message)
                
                if retry_count >= self.max_retries:
                    self.failed_calls += 1
                    logging.error("Max retries reached for API error, raising error")
                    raise
                
                # Exponential backoff for retrying
                delay = min(self.base_delay * (2 ** retry_count), self.max_delay)
                logging.info(f"API error. Retrying in {delay} seconds...")
                time.sleep(delay)
                
            except Exception as e:
                # Handle unexpected errors
                error_type = type(e).__name__
                error_message = str(e)
                
                # Track error type
                self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
                
                # Log and record the error
                logging.error(f"Unexpected error type {error_type}: {error_message[:100]}...")
                
                if self.tracker_manager:
                    self.tracker_manager.record_llm_error(current_url, error_type, error_message)
                
                self.failed_calls += 1
                raise
    
    def get_stats(self):
        """Get statistics about LLM calls"""
        success_rate = (self.successful_calls / self.total_calls * 100) if self.total_calls > 0 else 0
        avg_retries = sum(self.retry_counts) / len(self.retry_counts) if self.retry_counts else 0
        
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": f"{success_rate:.2f}%",
            "average_retries": avg_retries,
            "error_types": self.error_types
        }
