from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union, Any
import asyncio
import json
import logging

from .services.data_service import DataService
from .services.scraping_service import ScrapingService
from .services.llm_service import LLMService
from .services.visualization_service import VisualizationService
from .models import AnalysisRequest, AnalysisResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Data Analyst Agent",
    description="LLM-powered data analysis API that can scrape, process, analyze, and visualize data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
data_service = DataService()
scraping_service = ScrapingService()
llm_service = LLMService()
visualization_service = VisualizationService()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Data Analyst Agent is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "data_service": "available",
            "scraping_service": "available", 
            "llm_service": "available",
            "visualization_service": "available"
        }
    }

@app.post("/api/")
async def analyze_data(
    file: Optional[UploadFile] = File(None),
    data: Optional[str] = Form(None)
) -> List[Any]:
    """
    Main data analysis endpoint that accepts either file upload or form data
    
    Args:
        file: Optional uploaded file containing analysis task
        data: Optional form data containing analysis task
    
    Returns:
        List containing analysis results in the format specified by the task
    """
    try:
        # Extract task from file or form data
        if file:
            content = await file.read()
            task_content = content.decode('utf-8')
        elif data:
            task_content = data
        else:
            raise HTTPException(status_code=400, detail="No data provided")
        
        logger.info(f"Received analysis task: {task_content[:100]}...")
        
        # Parse the task
        try:
            # Try to parse as JSON first
            task_data = json.loads(task_content)
        except json.JSONDecodeError:
            # If not JSON, treat as plain text task
            task_data = {"task": task_content}
        
        # Process the analysis task
        results = await process_analysis_task(task_data)
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/analyze")
async def analyze_data_json(request: AnalysisRequest) -> AnalysisResponse:
    """
    JSON-based data analysis endpoint
    
    Args:
        request: Analysis request with task description and parameters
        
    Returns:
        Structured analysis response
    """
    try:
        logger.info(f"Processing JSON analysis request: {request.task[:100]}...")
        
        results = await process_analysis_task(request.dict())
        
        return AnalysisResponse(
            success=True,
            results=results,
            message="Analysis completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error in JSON analysis: {str(e)}")
        return AnalysisResponse(
            success=False,
            results=[],
            message=f"Analysis failed: {str(e)}"
        )

async def process_analysis_task(task_data: dict) -> List[Any]:
    """
    Process an analysis task and return results
    
    Args:
        task_data: Dictionary containing task description and parameters
        
    Returns:
        List of analysis results
    """
    task_description = task_data.get("task", "")
    questions = task_data.get("questions", [])
    url = task_data.get("url")
    query = task_data.get("query")
    
    results = []
    
    # Determine task type and process accordingly
    if "wikipedia" in task_description.lower() or "scrape" in task_description.lower():
        # Web scraping task
        if url:
            scraped_data = await scraping_service.scrape_url(url)
        else:
            # Extract Wikipedia URL from task or use default
            scraped_data = await scraping_service.scrape_wikipedia_films()
        
        # Process each question
        for question in questions:
            if "how many" in question.lower() and "$2" in question and "before 2000" in question:
                # Count $2bn movies before 2000
                count = await data_service.count_films_before_year(scraped_data, 2000, 2.0)
                results.append(count)
                
            elif "earliest film" in question.lower() and "$1.5" in question:
                # Find earliest film over $1.5bn
                film = await data_service.find_earliest_film_over_amount(scraped_data, 1.5)
                results.append(film)
                
            elif "correlation" in question.lower():
                # Calculate correlation
                correlation = await data_service.calculate_correlation(scraped_data, "Rank", "Peak")
                results.append(correlation)
                
            elif "scatterplot" in question.lower() or "plot" in question.lower():
                # Generate visualization
                chart_data = await data_service.prepare_chart_data(scraped_data, "Rank", "Peak")
                chart_base64 = await visualization_service.create_scatterplot_with_regression(
                    chart_data, "Rank", "Peak"
                )
                results.append(chart_base64)
    
    elif "database" in task_description.lower() or query:
        # Database query task
        if query:
            query_results = await data_service.execute_duckdb_query(query)
            results.extend(await process_database_results(query_results, questions))
    
    else:
        # General analysis task - use LLM
        llm_results = await llm_service.analyze_task(task_description, questions)
        results.extend(llm_results)
    
    return results

async def process_database_results(query_results: dict, questions: List[str]) -> List[Any]:
    """Process database query results based on questions"""
    results = []
    
    for question in questions:
        if "count" in question.lower():
            # Return count of records
            results.append(query_results.get("count", 0))
        elif "regression" in question.lower():
            # Calculate regression slope
            slope = await data_service.calculate_regression_slope(query_results)
            results.append(slope)
        elif "plot" in question.lower():
            # Generate plot
            chart_base64 = await visualization_service.create_plot_from_data(query_results)
            results.append(chart_base64)
        else:
            # Use LLM for complex questions
            llm_result = await llm_service.answer_question(question, query_results)
            results.append(llm_result)
    
    return results

@app.get("/docs")
async def get_docs():
    """Redirect to API documentation"""
    return {"message": "Visit /docs for API documentation"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
