import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import base64
import io
from typing import List, Dict, Any, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class VisualizationService:
    """Service for creating data visualizations and charts"""
    
    def __init__(self):
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
    
    async def create_scatterplot_with_regression(
        self, 
        data: List[Dict[str, Any]], 
        x_col: str, 
        y_col: str,
        title: Optional[str] = None
    ) -> str:
        """
        Create a scatterplot with regression line and return as base64 string
        
        Args:
            data: List of data dictionaries
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            title: Optional chart title
            
        Returns:
            Base64 encoded PNG image string
        """
        try:
            if not data:
                return self._create_error_chart("No data provided")
            
            df = pd.DataFrame(data)
            
            # Find the correct column names (case insensitive)
            x_column = self._find_column(df, x_col)
            y_column = self._find_column(df, y_col)
            
            if not x_column or not y_column:
                return self._create_error_chart(f"Columns '{x_col}' or '{y_col}' not found")
            
            # Convert to numeric
            x_data = pd.to_numeric(df[x_column], errors='coerce')
            y_data = pd.to_numeric(df[y_column], errors='coerce')
            
            # Remove NaN values
            valid_mask = ~(pd.isna(x_data) | pd.isna(y_data))
            x_clean = x_data[valid_mask]
            y_clean = y_data[valid_mask]
            
            if len(x_clean) < 2:
                return self._create_error_chart("Insufficient valid data points")
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Scatterplot
            ax.scatter(x_clean, y_clean, alpha=0.6, s=50, color='blue', edgecolors='black', linewidth=0.5)
            
            # Regression line
            if len(x_clean) >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
                line = slope * x_clean + intercept
                ax.plot(x_clean, line, color='red', linestyle='--', linewidth=2, label=f'Regression line (R² = {r_value**2:.3f})')
            
            # Formatting
            ax.set_xlabel(x_column.replace('_', ' ').title())
            ax.set_ylabel(y_column.replace('_', ' ').title())
            ax.set_title(title or f'{y_column.title()} vs {x_column.title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Convert to base64
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating scatterplot: {str(e)}")
            return self._create_error_chart(f"Error: {str(e)}")
    
    async def create_bar_chart(
        self,
        data: List[Dict[str, Any]],
        x_col: str,
        y_col: str,
        title: Optional[str] = None
    ) -> str:
        """Create a bar chart and return as base64 string"""
        try:
            if not data:
                return self._create_error_chart("No data provided")
            
            df = pd.DataFrame(data)
            
            x_column = self._find_column(df, x_col)
            y_column = self._find_column(df, y_col)
            
            if not x_column or not y_column:
                return self._create_error_chart(f"Columns '{x_col}' or '{y_col}' not found")
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Convert y to numeric
            y_data = pd.to_numeric(df[y_column], errors='coerce')
            
            # Create bar chart
            bars = ax.bar(df[x_column], y_data, color='skyblue', edgecolor='black', linewidth=0.5)
            
            # Formatting
            ax.set_xlabel(x_column.replace('_', ' ').title())
            ax.set_ylabel(y_column.replace('_', ' ').title())
            ax.set_title(title or f'{y_column.title()} by {x_column.title()}')
            
            # Rotate x-axis labels if needed
            if len(df) > 10:
                plt.xticks(rotation=45, ha='right')
            
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating bar chart: {str(e)}")
            return self._create_error_chart(f"Error: {str(e)}")
    
    async def create_line_chart(
        self,
        data: List[Dict[str, Any]],
        x_col: str,
        y_col: str,
        title: Optional[str] = None
    ) -> str:
        """Create a line chart and return as base64 string"""
        try:
            if not data:
                return self._create_error_chart("No data provided")
            
            df = pd.DataFrame(data)
            
            x_column = self._find_column(df, x_col)
            y_column = self._find_column(df, y_col)
            
            if not x_column or not y_column:
                return self._create_error_chart(f"Columns '{x_col}' or '{y_col}' not found")
            
            # Convert to numeric
            x_data = pd.to_numeric(df[x_column], errors='coerce')
            y_data = pd.to_numeric(df[y_column], errors='coerce')
            
            # Sort by x values
            sorted_indices = x_data.argsort()
            x_sorted = x_data.iloc[sorted_indices]
            y_sorted = y_data.iloc[sorted_indices]
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(x_sorted, y_sorted, marker='o', linewidth=2, markersize=6, color='blue')
            
            # Formatting
            ax.set_xlabel(x_column.replace('_', ' ').title())
            ax.set_ylabel(y_column.replace('_', ' ').title())
            ax.set_title(title or f'{y_column.title()} vs {x_column.title()}')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating line chart: {str(e)}")
            return self._create_error_chart(f"Error: {str(e)}")
    
    async def create_histogram(
        self,
        data: List[Dict[str, Any]],
        column: str,
        bins: int = 20,
        title: Optional[str] = None
    ) -> str:
        """Create a histogram and return as base64 string"""
        try:
            if not data:
                return self._create_error_chart("No data provided")
            
            df = pd.DataFrame(data)
            
            col_name = self._find_column(df, column)
            if not col_name:
                return self._create_error_chart(f"Column '{column}' not found")
            
            # Convert to numeric
            numeric_data = pd.to_numeric(df[col_name], errors='coerce').dropna()
            
            if len(numeric_data) == 0:
                return self._create_error_chart("No valid numeric data found")
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.hist(numeric_data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
            
            # Formatting
            ax.set_xlabel(col_name.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(title or f'Distribution of {col_name.title()}')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating histogram: {str(e)}")
            return self._create_error_chart(f"Error: {str(e)}")
    
    async def create_plot_from_data(self, query_results: Dict[str, Any]) -> str:
        """Create an appropriate plot from query results"""
        try:
            data = query_results.get("data", [])
            if not data:
                return self._create_error_chart("No data in query results")
            
            df = pd.DataFrame(data)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) >= 2:
                # Create scatterplot for two numeric columns
                return await self.create_scatterplot_with_regression(
                    data, 
                    numeric_cols[0], 
                    numeric_cols[1],
                    "Data Analysis Results"
                )
            elif len(numeric_cols) == 1:
                # Create histogram for single numeric column
                return await self.create_histogram(
                    data,
                    numeric_cols[0],
                    title="Data Distribution"
                )
            else:
                return self._create_error_chart("No numeric columns found for visualization")
                
        except Exception as e:
            logger.error(f"Error creating plot from data: {str(e)}")
            return self._create_error_chart(f"Error: {str(e)}")
    
    def _find_column(self, df: pd.DataFrame, target_col: str) -> Optional[str]:
        """Find column name in dataframe (case insensitive)"""
        target_lower = target_col.lower()
        
        # Exact match (case insensitive)
        for col in df.columns:
            if col.lower() == target_lower:
                return col
        
        # Partial match
        for col in df.columns:
            if target_lower in col.lower() or col.lower() in target_lower:
                return col
        
        return None
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            
            # Check size (aim for under 100KB)
            img_size = buffer.getbuffer().nbytes
            if img_size > 100000:  # 100KB
                # Reduce DPI and try again
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png', dpi=72, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                buffer.seek(0)
            
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close(fig)  # Free memory
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Error converting figure to base64: {str(e)}")
            plt.close(fig)
            return self._create_error_chart("Error generating image")
    
    def _create_error_chart(self, error_message: str) -> str:
        """Create a simple error message chart"""
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f"Error: {error_message}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14, color='red')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            return self._fig_to_base64(fig)
            
        except Exception:
            # If even error chart fails, return a minimal base64 encoded error image
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
Asset 9 of 15
