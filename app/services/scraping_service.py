import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict, Any, Optional
import asyncio
import logging
import re
from urllib.parse import urljoin, urlparse
import time

logger = logging.getLogger(__name__)

class ScrapingService:
    """Service for web scraping operations"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    async def scrape_url(self, url: str, selector: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Scrape data from a given URL
        
        Args:
            url: URL to scrape
            selector: Optional CSS selector for specific elements
            
        Returns:
            List of scraped data dictionaries
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            if selector:
                elements = soup.select(selector)
                return [{'text': elem.get_text(strip=True), 'html': str(elem)} for elem in elements]
            else:
                # Try to find tables automatically
                return await self._extract_tables_from_soup(soup)
                
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return []
    
    async def scrape_wikipedia_films(self, url: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Scrape Wikipedia highest grossing films data
        
        Args:
            url: Optional specific Wikipedia URL
            
        Returns:
            List of film data dictionaries
        """
        if not url:
            url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the main data table
            tables = soup.find_all('table', {'class': 'wikitable'})
            
            films_data = []
            
            for table in tables:
                # Check if this looks like the films table
                headers = [th.get_text(strip=True).lower() for th in table.find_all('th')]
                
                if any(keyword in ' '.join(headers) for keyword in ['rank', 'film', 'worldwide', 'gross']):
                    films_data.extend(await self._parse_films_table(table))
            
            return films_data
            
        except Exception as e:
            logger.error(f"Error scraping Wikipedia films: {str(e)}")
            return []
    
    async def scrape_wikipedia_table(self, url: str, table_index: int = 0) -> List[Dict[str, Any]]:
        """
        Scrape a specific table from Wikipedia
        
        Args:
            url: Wikipedia page URL
            table_index: Index of table to scrape (0-based)
            
        Returns:
            List of table row dictionaries
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Use pandas for easier table parsing
            tables = pd.read_html(response.content)
            
            if table_index < len(tables):
                df = tables[table_index]
                return df.to_dict('records')
            else:
                logger.warning(f"Table index {table_index} not found, found {len(tables)} tables")
                return []
                
        except Exception as e:
            logger.error(f"Error scraping Wikipedia table: {str(e)}")
            # Fallback to BeautifulSoup method
            return await self._scrape_table_with_bs4(url, table_index)
    
    async def _extract_tables_from_soup(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract all tables from soup object"""
        tables_data = []
        
        tables = soup.find_all('table')
        
        for table in tables:
            table_data = await self._parse_generic_table(table)
            if table_data:
                tables_data.extend(table_data)
        
        return tables_data
    
    async def _parse_films_table(self, table) -> List[Dict[str, Any]]:
        """Parse Wikipedia films table specifically"""
        films = []
        
        # Get headers
        header_row = table.find('tr')
        headers = []
        
        if header_row:
            for th in header_row.find_all(['th', 'td']):
                header_text = th.get_text(strip=True)
                headers.append(self._normalize_header(header_text))
        
        # Get data rows
        rows = table.find_all('tr')[1:]  # Skip header row
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            
            if len(cells) >= 3:  # Must have at least rank, title, gross
                film_data = {}
                
                for i, cell in enumerate(cells):
                    if i < len(headers):
                        header = headers[i]
                        cell_text = cell.get_text(strip=True)
                        
                        # Clean and parse cell data
                        if header == 'rank':
                            film_data['rank'] = self._parse_number(cell_text)
                        elif header in ['title', 'film']:
                            film_data['title'] = self._clean_film_title(cell_text)
                        elif 'gross' in header or 'revenue' in header:
                            film_data['worldwide_gross'] = cell_text
                        elif header == 'year':
                            film_data['year'] = self._parse_year(cell_text)
                        elif header == 'peak':
                            film_data['peak'] = self._parse_number(cell_text)
                        else:
                            film_data[header] = cell_text
                
                if 'title' in film_data and film_data['title']:
                    films.append(film_data)
        
        return films
    
    async def _parse_generic_table(self, table) -> List[Dict[str, Any]]:
        """Parse any generic table"""
        data = []
        
        # Get headers
        header_row = table.find('tr')
        headers = []
        
        if header_row:
            for th in header_row.find_all(['th', 'td']):
                header_text = th.get_text(strip=True)
                headers.append(self._normalize_header(header_text))
        
        # Get data rows
        rows = table.find_all('tr')[1:]  # Skip header row
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            
            if cells:
                row_data = {}
                
                for i, cell in enumerate(cells):
                    if i < len(headers):
                        header = headers[i] if headers else f'column_{i}'
                        cell_text = cell.get_text(strip=True)
                        row_data[header] = cell_text
                
                if any(row_data.values()):  # Only add non-empty rows
                    data.append(row_data)
        
        return data
    
    async def _scrape_table_with_bs4(self, url: str, table_index: int) -> List[Dict[str, Any]]:
        """Fallback method using BeautifulSoup"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            tables = soup.find_all('table')
            
            if table_index < len(tables):
                return await self._parse_generic_table(tables[table_index])
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error in BeautifulSoup fallback: {str(e)}")
            return []
    
    def _normalize_header(self, header: str) -> str:
        """Normalize table header names"""
        header = header.lower().strip()
        
        # Common normalizations
        normalizations = {
            'worldwide gross': 'worldwide_gross',
            'box office': 'worldwide_gross',
            'total gross': 'worldwide_gross',
            'film title': 'title',
            'movie': 'title',
            'release year': 'year',
            'year released': 'year',
        }
        
        for key, value in normalizations.items():
            if key in header:
                return value
        
        # Clean up header name
        header = re.sub(r'[^\w\s]', '', header)  # Remove special chars
        header = re.sub(r'\s+', '_', header)     # Replace spaces with underscores
        
        return header
    
    def _clean_film_title(self, title: str) -> str:
        """Clean film title text"""
        # Remove citations and extra info
        title = re.sub(r'\[.*?\]', '', title)  # Remove [1], [citation needed], etc.
        title = re.sub(r'\(.*?\)', '', title)  # Remove (2009), (film), etc.
        return title.strip()
    
    def _parse_number(self, text: str) -> Optional[int]:
        """Extract integer from text"""
        numbers = re.findall(r'\d+', text)
        return int(numbers[0]) if numbers else None
    
    def _parse_year(self, text: str) -> Optional[int]:
        """Extract year from text"""
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        return int(year_match.group()) if year_match else None
