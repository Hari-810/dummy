import os
import json
import re
import pandas as pd
from io import StringIO
from base_generator import BaseSyntheticGenerator
from langchain_openai import AzureChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage
from constants import BATCH_SIZE,TABULAR_DATA_FOLDER
import numpy as np
from langchain_core.output_parsers import JsonOutputParser
import csv
# Tabular Data Generator
class TabularDataGenerator(BaseSyntheticGenerator):
    column_names = ""
    def __init__(self):
        super().__init__(data_category="tabular")

    def sanitize_filename(self,query: str, max_length: int = 50) -> str:
        """
        Converts a user query into a valid filename by removing special characters, 
        replacing spaces with underscores, and truncating if necessary.

        Args:
            query (str): User query string.
            max_length (int): Maximum allowed length for the filename.

        Returns:
            str: Sanitized filename.
        """
        match = re.search(r'for\s+(.+)', query, re.IGNORECASE)
        filename = match.group(1) if match else query  
        filename = re.sub(r'[^a-zA-Z0-9\s]', '', filename)  
        filename = re.sub(r'\s+', '_', filename)  
        return filename[:max_length] + ".csv"

    def acquire_user_inputs(self):
            """Acquires user inputs for synthetic data generation."""
            user_query = input("\nEnter your query for synthetic data generation: ").strip()
            num_samples = int(input("Enter the number of synthetic samples to generate: "))
            realism_level = input("Enter realism level (High-Fidelity, Noisy, Fully Randomized): ").strip()
            self.column_names = input("Enter column names (comma-separated, or leave empty for automatic inference): ").strip() or None
            domain = input("Enter the domain (e.g., Finance, Healthcare, etc.): ").strip()
            feature_types = input("Enter feature types (Categorical, Numerical, Mixed): ").strip()
            feature_distributions = input("Enter feature distributions (Normal, Uniform, Poisson, Custom): ").strip()
            relationships_constraints = input("Enter relationships/constraints (e.g., Correlations, Logical Constraints): ").strip()
            output_format = input("Enter output format (CSV, JSON, Parquet, SQL): ").strip()
            schema_input = input("Enter schema details (JSON formatted): ").strip()
            schema_details = json.loads(schema_input)

            return {
                'user_query': user_query,
                'num_samples': num_samples,
                'realism_level': realism_level,
                'column_names': self.column_names,
                'domain': domain,
                'feature_types': feature_types,
                'feature_distributions': feature_distributions,
                'relationships_constraints': relationships_constraints,
                'output_format': output_format,
                'schema_details': schema_details
            }

    def generate_synthetic_data(self,user_query: str, llm, num_samples: int, 
                            realism_level: str, column_names: str, domain: str, feature_types: str, 
                            feature_distributions: str, relationships_constraints: str, 
                            output_format: str, schema_details: dict):
        """
        Generates synthetic data using Azure OpenAI LLM without batch processing.
        """
        generated_samples = []
        parser = JsonOutputParser()
        
        self.prompt_template = PromptTemplate(
            template=self.prompt_template["synthetic_data_generation_prompt"],
            input_variables=["user_request", "num_samples", "realism_level", "column_names", "domain",
                            "feature_types", "feature_distributions", "relationships_constraints", "output_format", "schema_details"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        
        formatted_prompt = self.prompt_template.template.format(
            user_request=str(user_query),
            llm=self.llm_model,
            num_samples=str(num_samples),
            realism_level=str(realism_level).strip(), 
            column_names=str(column_names),
            domain=str(domain),
            feature_types=str(feature_types),
            feature_distributions=str(feature_distributions),
            relationships_constraints=str(relationships_constraints),
            output_format=str(output_format),
            schema_details=json.dumps(schema_details)  
        )
        
        try:
            response = self.llm_model.invoke([HumanMessage(content=formatted_prompt)])
            print("------------> Raw LLM Response:", response)
            
            if response and hasattr(response, "content"):
                print("The LLM response is:", response.content)

                try:
                    # Extract JSON content using regex
                    json_match = re.search(r'```json\n(.*?)\n```', response.content, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1)
                    else:
                        json_content = response  # If no markdown-style JSON, assume raw JSON
                    
                    # Parse JSON content
                    parsed_data = json.loads(json_content)
                    
                    # Convert to DataFrame
                    df =  pd.DataFrame(parsed_data)
                   
                    print("LLm response parsed successfully.")
                    print(df.head())
                    
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error parsing JSON: {e}")
        except Exception as e:
            print(f"Error generating synthetic data: {e}")
            return None
      
        return df
        
    def save_output(self, data, output_file="synthetic_data.csv"):
        super().save_output(data,output_file)
