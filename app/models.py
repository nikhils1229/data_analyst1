from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict, Union
from datetime import datetime

class AnalysisRequest(BaseModel):
    """Request model for data analysis tasks"""
    task: str = Field(..., description="Description of the analysis task to perform")
    questions: Optional[List[str]] = Field(default=[], description="Specific questions to answer")
    url: Optional[str] = Field(default=None, description="URL to scrape data from")
    query: Optional[str] = Field(default=None, description="Database query to execute")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Additional parameters")

class AnalysisResponse(BaseModel):
    """Response model for analysis results"""
    success: bool = Field(..., description="Whether the analysis was successful")
    results: List[Any] = Field(..., description="Analysis results")
    message: str = Field(..., description="Status message")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Response timestamp")

class ScrapingRequest(BaseModel):
    """Request model for web scraping tasks"""
    url: str = Field(..., description="URL to scrape")
    selector: Optional[str] = Field(default=None, description="CSS selector for specific elements")
    data_type: Optional[str] = Field(default="table", description="Type of data to extract")

class ChartRequest(BaseModel):
    """Request model for chart generation"""
    data: List[Dict[str, Any]] = Field(..., description="Data to visualize")
    chart_type: str = Field(..., description="Type of chart to generate")
    x_column: str = Field(..., description="Column for x-axis")
    y_column: str = Field(..., description="Column for y-axis")
    title: Optional[str] = Field(default="", description="Chart title")
    
class DatabaseQueryRequest(BaseModel):
    """Request model for database queries"""
    query: str = Field(..., description="SQL query to execute")
    database_url: Optional[str] = Field(default=None, description="Database connection URL")
    
class LLMAnalysisRequest(BaseModel):
    """Request model for LLM-based analysis"""
    task: str = Field(..., description="Analysis task description")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Data context for analysis")
    model: Optional[str] = Field(default="gpt-4", description="LLM model to use")
    
class FilmData(BaseModel):
    """Model for film data from Wikipedia scraping"""
    rank: Optional[int] = None
    title: str
    worldwide_gross: Optional[float] = None
    year: Optional[int] = None
    peak: Optional[int] = None
    
class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
