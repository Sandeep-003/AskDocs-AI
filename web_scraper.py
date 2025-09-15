"""
Web scraping utilities for documentation.
Handles scraping of documentation websites.
"""

import time
import logging
import requests
from pathlib import Path
from typing import Set, Optional
from urllib.parse import urljoin, urlparse
from collections import deque
from bs4 import BeautifulSoup

from config import config

# Set up logging
logger = logging.getLogger(__name__)


class WebScraper:
    """Web scraper for documentation sites."""
    
    def __init__(self, output_folder: str):
        """Initialize scraper with output folder."""
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        
        # Set up session with headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def _is_valid_url(self, url: str, base_netloc: str) -> bool:
        """Check if URL is valid for scraping."""
        try:
            parsed = urlparse(url)
            return (
                parsed.scheme in ('http', 'https') and 
                parsed.netloc == base_netloc
            )
        except Exception:
            return False
    
    def _get_all_links(self, soup: BeautifulSoup, base_url: str, base_netloc: str) -> Set[str]:
        """Extract all valid links from a page."""
        links = set()
        
        for a_tag in soup.find_all('a', href=True):
            try:
                href = urljoin(base_url, a_tag['href'])
                if self._is_valid_url(href, base_netloc):
                    links.add(href)
            except Exception as e:
                logger.debug(f"Failed to process link {a_tag.get('href')}: {e}")
        
        return links
    
    def _save_content(self, content: str, url: str, file_type: str = 'txt') -> None:
        """Save content to file."""
        try:
            parsed = urlparse(url)
            path = parsed.path.strip('/')
            
            if not path:
                path = 'index'
            
            # Clean filename
            filename = path.replace('/', '_').replace('?', '_').replace('#', '_')
            filename = f"{filename}.{file_type}"
            
            filepath = self.output_folder / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.debug(f"Saved content to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save content for {url}: {e}")
    
    def scrape_website(self, home_url: str, max_pages: Optional[int] = None) -> None:
        """Scrape entire website starting from home URL."""
        visited = set()
        queue = deque([home_url])
        base_netloc = urlparse(home_url).netloc
        pages_scraped = 0
        
        logger.info(f"Starting website scrape from: {home_url}")
        
        while queue and (max_pages is None or pages_scraped < max_pages):
            url = queue.popleft()
            
            if url in visited:
                continue
            
            try:
                # Add delay between requests
                time.sleep(config.scraping.delay_between_requests)
                
                # Make request with timeout
                response = self.session.get(url, timeout=config.scraping.timeout)
                response.raise_for_status()
                
                # Parse content
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract text content
                text = soup.get_text(separator='\\n', strip=True)
                
                if text.strip():
                    self._save_content(text, url)
                    pages_scraped += 1
                    logger.info(f"Scraped ({pages_scraped}): {url}")
                
                # Find new links
                links = self._get_all_links(soup, url, base_netloc)
                for link in links:
                    if link not in visited:
                        queue.append(link)
                
                visited.add(url)
                
            except requests.RequestException as e:
                logger.error(f"Request failed for {url}: {e}")
            except Exception as e:
                logger.error(f"Failed to process {url}: {e}")
        
        logger.info(f"Scraping completed. Pages scraped: {pages_scraped}")