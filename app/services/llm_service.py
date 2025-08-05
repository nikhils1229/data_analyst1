import os
from typing import List, Dict, Any, Optional
import asyncio
import logging
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import json

logger = logging.getLogger(__name__)

class LLMService:
    """Service for LLM integration and AI-powered analysis"""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        
        # Initialize clients if API keys are available
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.openai_client = AsyncOpenAI(api_key=openai_key)
        
        anthropic_key = os.getenv("ANTHROPIC_API_KEY") 
        if anthropic_key:
            self.anthropic_client = AsyncAnthropic(api_key=anthropic_key)
    
    async def analyze_task(self, task_description: str, questions: List[str]) -> List[Any]:
        """
        Use LLM to analyze a task and provide answers
        
        Args:
            task_description: Description of the analysis task
            questions: List of questions to answer
            
        Returns:
            List of answers
        """
        try:
            if not self.openai_client and not self.anthropic_client:
                logger.warning("No LLM clients available")
                return ["LLM service not configured"] * len(questions)
            
            answers = []
            
            for question in questions:
                answer = await self._answer_single_question(task_description, question)
                answers.append(answer)
            
            return answers
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            return [f"Error: {str(e)}"] * len(questions)
    
    async def answer_question(self, question: str, context_data: Dict[str, Any]) -> str:
        """
        Answer a specific question given context data
        
        Args:
            question: Question to answer
            context_data: Context data for the question
            
        Returns:
            Answer string
        """
        try:
            # Format context data
            context_str = self._format_context_data(context_data)
            
            prompt = f"""
            Given the following data context:
            {context_str}
            
            Please answer this question: {question}
            
            Provide a clear, concise answer based on the data provided.
            """
            
            if self.openai_client:
                return await self._query_openai(prompt)
            elif self.anthropic_client:
                return await self._query_anthropic(prompt)
            else:
                return "LLM service not available"
                
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"Error: {str(e)}"
    
    async def generate_analysis_summary(self, data: List[Dict[str, Any]], analysis_type: str) -> str:
        """
        Generate a summary analysis of data
        
        Args:
            data: Data to analyze
            analysis_type: Type of analysis to perform
            
        Returns:
            Analysis summary
        """
        try:
            # Sample data if too large
            sample_data = data[:100] if len(data) > 100 else data
            
            prompt = f"""
            Please perform a {analysis_type} analysis on the following dataset:
            
            Data sample (first {len(sample_data)} rows):
            {json.dumps(sample_data, indent=2)}
            
            Total dataset size: {len(data)} rows
            
            Provide insights, patterns, and key findings from this data.
            """
            
            if self.openai_client:
                return await self._query_openai(prompt)
            elif self.anthropic_client:
                return await self._query_anthropic(prompt)
            else:
                return "LLM service not available"
                
        except Exception as e:
            logger.error(f"Error generating analysis summary: {str(e)}")
            return f"Error: {str(e)}"
    
    async def extract_data_insights(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract key insights from data using LLM
        
        Args:
            data: Data to analyze
            
        Returns:
            Dictionary of insights
        """
        try:
            # Basic statistics
            import pandas as pd
            df = pd.DataFrame(data)
            
            stats_summary = {
                "total_records": len(df),
                "columns": list(df.columns),
                "numeric_columns": list(df.select_dtypes(include=['number']).columns),
                "missing_values": df.isnull().sum().to_dict()
            }
            
            prompt = f"""
            Analyze this dataset and provide key insights:
            
            Dataset Statistics:
            {json.dumps(stats_summary, indent=2)}
            
            Sample Data:
            {json.dumps(data[:10], indent=2)}
            
            Please provide:
            1. Key patterns or trends
            2. Data quality observations
            3. Potential areas for deeper analysis
            4. Recommendations for visualization
            
            Format your response as JSON with these keys: patterns, quality, analysis_suggestions, viz_recommendations
            """
            
            if self.openai_client:
                response = await self._query_openai(prompt)
            elif self.anthropic_client:
                response = await self._query_anthropic(prompt)
            else:
                return {"error": "LLM service not available"}
            
            # Try to parse as JSON, fallback to structured text
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {"analysis": response}
                
        except Exception as e:
            logger.error(f"Error extracting insights: {str(e)}")
            return {"error": str(e)}
    
    async def _answer_single_question(self, task_description: str, question: str) -> str:
        """Answer a single question given task context"""
        try:
            prompt = f"""
            Task: {task_description}
            
            Question: {question}
            
            Based on the task description, please provide the best answer you can to this question.
            If the question requires specific data analysis, indicate what type of data would be needed.
            """
            
            if self.openai_client:
                return await self._query_openai(prompt)
            elif self.anthropic_client:
                return await self._query_anthropic(prompt)
            else:
                return "Unable to process question - LLM service not available"
                
        except Exception as e:
            logger.error(f"Error answering single question: {str(e)}")
            return f"Error: {str(e)}"
    
    async def _query_openai(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """Query OpenAI API"""
        try:
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise e
    
    async def _query_anthropic(self, prompt: str, model: str = "claude-3-haiku-20240307") -> str:
        """Query Anthropic API"""
        try:
            response = await self.anthropic_client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise e
    
    def _format_context_data(self, context_data: Dict[str, Any]) -> str:
        """Format context data for LLM prompt"""
        try:
            if isinstance(context_data, dict):
                # Handle different data structures
                if "data" in context_data:
                    data = context_data["data"]
                    if isinstance(data, list) and len(data) > 0:
                        # Show sample of data
                        sample = data[:5] if len(data) > 5 else data
                        return f"Data sample ({len(data)} total records):\n{json.dumps(sample, indent=2)}"
                    else:
                        return json.dumps(context_data, indent=2)
                else:
                    return json.dumps(context_data, indent=2)
            else:
                return str(context_data)
                
        except Exception as e:
            logger.error(f"Error formatting context data: {str(e)}")
            return "Error formatting context data"
