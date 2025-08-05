import pandas as pd
import numpy as np
import duckdb
from typing import List, Dict, Any, Optional
import asyncio
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class DataService:
    """Service for data processing, analysis, and database operations"""
    
    def __init__(self):
        self.conn = duckdb.connect()
        
    async def count_films_before_year(self, data: List[Dict], year: int, min_gross: float) -> int:
        """
        Count films released before a specific year with minimum gross revenue
        
        Args:
            data: List of film data dictionaries
            year: Year threshold
            min_gross: Minimum gross revenue in billions
            
        Returns:
            Count of matching films
        """
        try:
            df = pd.DataFrame(data)
            
            # Clean and convert revenue data
            if 'worldwide_gross' in df.columns:
                df['gross_billions'] = df['worldwide_gross'].apply(self._parse_revenue_to_billions)
            elif 'revenue' in df.columns:
                df['gross_billions'] = df['revenue'].apply(self._parse_revenue_to_billions)
            else:
                # Try to find revenue column
                revenue_cols = [col for col in df.columns if any(keyword in col.lower() 
                               for keyword in ['gross', 'revenue', 'box office'])]
                if revenue_cols:
                    df['gross_billions'] = df[revenue_cols[0]].apply(self._parse_revenue_to_billions)
                else:
                    return 0
            
            # Parse year from release date or year column
            if 'year' in df.columns:
                df['release_year'] = pd.to_numeric(df['year'], errors='coerce')
            elif 'release_date' in df.columns:
                df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
            else:
                # Try to extract year from title or other columns
                df['release_year'] = df.apply(self._extract_year_from_row, axis=1)
            
            # Filter and count
            filtered = df[
                (df['release_year'] < year) & 
                (df['gross_billions'] >= min_gross) & 
                (pd.notna(df['release_year'])) & 
                (pd.notna(df['gross_billions']))
            ]
            
            return len(filtered)
            
        except Exception as e:
            logger.error(f"Error counting films: {str(e)}")
            return 0
    
    async def find_earliest_film_over_amount(self, data: List[Dict], min_gross: float) -> str:
        """
        Find the earliest film that grossed over a specified amount
        
        Args:
            data: List of film data dictionaries  
            min_gross: Minimum gross revenue in billions
            
        Returns:
            Title of the earliest film
        """
        try:
            df = pd.DataFrame(data)
            
            # Clean and convert revenue data
            if 'worldwide_gross' in df.columns:
                df['gross_billions'] = df['worldwide_gross'].apply(self._parse_revenue_to_billions)
            else:
                revenue_cols = [col for col in df.columns if any(keyword in col.lower() 
                               for keyword in ['gross', 'revenue', 'box office'])]
                if revenue_cols:
                    df['gross_billions'] = df[revenue_cols[0]].apply(self._parse_revenue_to_billions)
                else:
                    return "Unknown"
            
            # Parse year
            if 'year' in df.columns:
                df['release_year'] = pd.to_numeric(df['year'], errors='coerce')
            else:
                df['release_year'] = df.apply(self._extract_year_from_row, axis=1)
            
            # Filter films over the amount
            filtered = df[
                (df['gross_billions'] >= min_gross) & 
                (pd.notna(df['release_year'])) & 
                (pd.notna(df['gross_billions']))
            ]
            
            if len(filtered) == 0:
                return "No films found"
            
            # Find earliest
            earliest = filtered.loc[filtered['release_year'].idxmin()]
            return earliest.get('title', 'Unknown')
            
        except Exception as e:
            logger.error(f"Error finding earliest film: {str(e)}")
            return "Error"
    
    async def calculate_correlation(self, data: List[Dict], col1: str, col2: str) -> float:
        """
        Calculate correlation between two numeric columns
        
        Args:
            data: List of data dictionaries
            col1: First column name
            col2: Second column name
            
        Returns:
            Correlation coefficient
        """
        try:
            df = pd.DataFrame(data)
            
            # Clean column names (case insensitive matching)
            df.columns = df.columns.str.lower()
            col1_clean = col1.lower()
            col2_clean = col2.lower()
            
            if col1_clean not in df.columns or col2_clean not in df.columns:
                # Try to find similar column names
                col1_match = self._find_similar_column(df.columns, col1_clean)
                col2_match = self._find_similar_column(df.columns, col2_clean)
                
                if col1_match and col2_match:
                    col1_clean = col1_match
                    col2_clean = col2_match
                else:
                    return 0.0
            
            # Convert to numeric
            series1 = pd.to_numeric(df[col1_clean], errors='coerce')
            series2 = pd.to_numeric(df[col2_clean], errors='coerce')
            
            # Calculate correlation
            correlation = series1.corr(series2)
            
            return round(correlation, 6) if pd.notna(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {str(e)}")
            return 0.0
    
    async def prepare_chart_data(self, data: List[Dict], x_col: str, y_col: str) -> List[Dict]:
        """
        Prepare data for chart visualization
        
        Args:
            data: Raw data
            x_col: X-axis column name
            y_col: Y-axis column name
            
        Returns:
            Cleaned data for charting
        """
        try:
            df = pd.DataFrame(data)
            
            # Clean column names
            df.columns = df.columns.str.lower()
            x_col_clean = x_col.lower()
            y_col_clean = y_col.lower()
            
            # Find matching columns
            x_match = self._find_similar_column(df.columns, x_col_clean)
            y_match = self._find_similar_column(df.columns, y_col_clean)
            
            if not x_match or not y_match:
                return []
            
            # Extract and clean data
            chart_df = df[[x_match, y_match]].copy()
            chart_df[x_match] = pd.to_numeric(chart_df[x_match], errors='coerce')
            chart_df[y_match] = pd.to_numeric(chart_df[y_match], errors='coerce')
            
            # Remove NaN values
            chart_df = chart_df.dropna()
            
            return chart_df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error preparing chart data: {str(e)}")
            return []
    
    async def execute_duckdb_query(self, query: str) -> Dict[str, Any]:
        """
        Execute a DuckDB query
        
        Args:
            query: SQL query string
            
        Returns:
            Query results as dictionary
        """
        try:
            result = self.conn.execute(query).fetchdf()
            return {
                "data": result.to_dict('records'),
                "columns": list(result.columns),
                "count": len(result)
            }
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return {"data": [], "columns": [], "count": 0, "error": str(e)}
    
    async def calculate_regression_slope(self, query_results: Dict[str, Any]) -> float:
        """Calculate regression slope from query results"""
        try:
            df = pd.DataFrame(query_results.get("data", []))
            if len(df) < 2:
                return 0.0
            
            # Find numeric columns for regression
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return 0.0
            
            x = df[numeric_cols[0]]
            y = df[numeric_cols[1]]
            
            # Calculate slope using least squares
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
            
        except Exception as e:
            logger.error(f"Error calculating regression slope: {str(e)}")
            return 0.0
    
    def _parse_revenue_to_billions(self, revenue_str: str) -> Optional[float]:
        """Parse revenue string to billions (float)"""
        if pd.isna(revenue_str) or revenue_str == '':
            return None
            
        try:
            # Convert to string and clean
            revenue_str = str(revenue_str).replace(',', '').replace('$', '')
            
            # Extract numeric value
            numbers = re.findall(r'[\d.]+', revenue_str)
            if not numbers:
                return None
                
            value = float(numbers[0])
            
            # Convert to billions based on scale indicators
            if 'billion' in revenue_str.lower() or 'b' in revenue_str.lower():
                return value
            elif 'million' in revenue_str.lower() or 'm' in revenue_str.lower():
                return value / 1000
            elif 'trillion' in revenue_str.lower() or 't' in revenue_str.lower():
                return value * 1000
            else:
                # Assume the value is already in appropriate scale
                if value > 1000:  # Likely millions
                    return value / 1000
                else:
                    return value
                    
        except Exception as e:
            logger.error(f"Error parsing revenue '{revenue_str}': {str(e)}")
            return None
    
    def _extract_year_from_row(self, row: pd.Series) -> Optional[int]:
        """Extract year from any column in the row"""
        for value in row.values:
            if pd.isna(value):
                continue
                
            # Look for 4-digit year in the value
            year_match = re.search(r'\b(19|20)\d{2}\b', str(value))
            if year_match:
                return int(year_match.group())
        
        return None
    
    def _find_similar_column(self, columns: pd.Index, target: str) -> Optional[str]:
        """Find column name that matches target (fuzzy matching)"""
        target = target.lower()
        
        # Exact match
        if target in columns:
            return target
            
        # Partial match
        for col in columns:
            if target in col.lower() or col.lower() in target:
                return col
                
        # Look for common synonyms
        synonyms = {
            'rank': ['position', 'place', '#'],
            'peak': ['max', 'maximum', 'highest', 'top'],
            'gross': ['revenue', 'earnings', 'box office'],
            'year': ['date', 'released', 'release']
        }
        
        for synonym_list in synonyms.values():
            if target in synonym_list:
                for col in columns:
                    if any(syn in col.lower() for syn in synonym_list):
                        return col
        
        return None
