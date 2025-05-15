import json
import re
import logging
import pandas as pd
import time
from hashlib import sha256
from .base_generator import BaseSyntheticGenerator
from .token_estimator import TokenEstimator
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage
from config.constants import GENERIC_VARIABLES, TDC_VARIABLES, DATA_CATEGORY,MAX_RETRIES

class TabularDataGenerator(BaseSyntheticGenerator):
    """
    A generator class for producing synthetic tabular data using LLM prompts.
    Extends the BaseSyntheticGenerator with tabular-specific logic, including
    adaptive batch sizing based on token estimation.
    """
    column_names = ""
    def __init__(self):
        """
        Initializes the TabularDataGenerator class.
        - Calls the parent class constructor with the data category set to "tabular".
        - Sets up logging for this class instance.
        - Initializes the TokenEstimator for adaptive batching.
        - Initializes a parser for interpreting structured JSON outputs from the LLM.
        """
        super().__init__(data_category=DATA_CATEGORY.TABULAR.value)
        self.logger = logging.getLogger(__name__)
        self.token_estimator = TokenEstimator()
        self.parser = JsonOutputParser()

    def extract_json_from_response(self, response_text: str):
        """
        Safely extracts JSON from an LLM response even if it's wrapped in Markdown or contains additional text.
        Detects truncation and logs appropriately.
        """
        try:
            # First: try to extract Markdown-style JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Fallback: try to extract direct JSON array
                json_match = re.search(r'(\[\s*{.*?}\s*\])', response_text, re.DOTALL)
                json_str = json_match.group(1) if json_match else response_text.strip()

            # Optional: basic truncation check
            if not json_str.strip().endswith(("]", "}")):
                self.logger.warning("Possible output truncation detected: JSON does not end with expected character.")

            # Optional: attempt to auto-close the array (only if it *looks* like it's truncated)
            if json_str.count("{") > json_str.count("}"):
                self.logger.warning("Attempting to fix incomplete JSON object.")
                json_str += "}" * (json_str.count("{") - json_str.count("}"))

            if json_str.count("[") > json_str.count("]"):
                self.logger.warning("Attempting to fix incomplete JSON array.")
                json_str += "]" * (json_str.count("[") - json_str.count("]"))

            return json.loads(json_str)

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed: {e}")
            self.logger.error(f"Failed content: {json_str[:500]}...")
            return [] 
        
    def generate_synthetic_data(self, data_category: str, user_query: str, num_samples: int,
                                realism_level: str, column_names: str, field_description: str, max_length: int,
                                domain: str, feature_types: str, feature_distributions: str,
                                relationships_constraints: str, output_format: str, schema_details: dict):
        """
         Generates synthetic tabular data using a language model based on the provided schema and generation parameters.

        This method handles:
        - Prompt template initialization.
        - Token estimation for schema and prompt.
        - Adaptive batch size calculation based on token limits.
        - Batched generation and parsing of results.
        - Deduplication and metrics tracking.

        Args:
            data_category (str): The category of data being generated (e.g., 'tabular').
            user_query (str): A user-provided description of the data to be generated.
            num_samples (int): Total number of synthetic records to generate.
            realism_level (str): The desired level of realism for the generated data.
            column_names (str): Column names of the tabular data.
            field_description (str): Descriptions of the fields (columns).
            max_length (int): Maximum character length allowed for field values.
            domain (str): The domain context for the data (e.g., finance, healthcare).
            feature_types (str): Types of features (e.g., categorical, numerical).
            feature_distributions (str): Distribution types of features (e.g., normal, uniform).
            relationships_constraints (str): Constraints or relationships between fields.
            output_format (str): The desired output format (e.g., JSON, CSV).
            schema_details (dict): Detailed schema metadata including column info and constraints.

        Returns:
            list: A list of generated synthetic records formatted according to the output format.
        """
        self.initialize_prompt_template(column_names, field_description, max_length,domain, feature_types, 
                                        feature_distributions, relationships_constraints, schema_details)
        all_data, seen_rows = [], set()
        metrics = self.initialize_metrics()

        # Estimate tokens
        schema_tokens = self.token_estimator.estimate_tokens_for_schema(schema_details)
        self.token_estimator.log_token_usage(json.dumps(schema_details), label="Schema")
        prompt_sample = self.build_prompt(user_query, 1, realism_level, column_names,
                                          field_description, max_length, domain, feature_types,
                                          feature_distributions, relationships_constraints, output_format,
                                          schema_details)
        prompt_tokens = self.token_estimator.log_token_usage(prompt_sample, label="Prompt Sample")

        dynamic_batch_size = self.token_estimator.get_dynamic_batch_size(schema_tokens, prompt_tokens)
        self.logger.info(f"Using adaptive batch size: {dynamic_batch_size}")

        for i in range(0, num_samples, dynamic_batch_size):
            batch_size = min(dynamic_batch_size, num_samples - i)
            self.logger.info(f"Processing batch {i // dynamic_batch_size + 1} with {batch_size} records")

            prompt = self.build_prompt(user_query, batch_size, realism_level, column_names,
                                       field_description, max_length, domain, feature_types,
                                       feature_distributions, relationships_constraints, output_format,
                                       schema_details)

            batch_data, batch_metrics = self.generate_batch(prompt, seen_rows, i, batch_size)
            all_data.extend(batch_data)
            self.update_metrics(metrics, batch_metrics)

        return self.finalize_generation(all_data, metrics)
    
    def initialize_prompt_template(self, column_names, field_description, max_length,
                               domain, feature_types, feature_distributions, relationships_constraints, schema_details):
        """
        Initializes the prompt template and context for tabular data generation.
        This method prepares the reusable prompt template that will be used to generate LLM prompts.
        It sets up the structure of the prompt using the input variables, format instructions,
        and user-provided schema context.

        Args:
            column_names (str): Names of the columns in the tabular data.
            field_description (str): Description for each column/field.
            max_length (int): Maximum allowed character length per field.
            domain (str): Domain context of the dataset (e.g., retail, finance).
            feature_types (str): Types of features (e.g., numerical, categorical).
            feature_distributions (str): Statistical distributions expected in the features.
            relationships_constraints (str): Logical or statistical relationships among features.
            schema_details (dict): Full schema metadata including constraints and descriptions.

        Returns:
            None
        """
        self.prompt_template = PromptTemplate(
            template=self.prompt_template[TDC_VARIABLES.YAML_FILE_NAME],
            input_variables=[
                TDC_VARIABLES.USER_REQUEST, GENERIC_VARIABLES.SAMPLE_COUNT, GENERIC_VARIABLES.REALISM_LEVEL,
                TDC_VARIABLES.COLUMN_NAMES, TDC_VARIABLES.FIELD_DESCRIPTION, TDC_VARIABLES.MAX_LENGTH,
                GENERIC_VARIABLES.DOMAIN, TDC_VARIABLES.FEATURE_TYPES, TDC_VARIABLES.FEATURE_DISTRIBUTION,
                TDC_VARIABLES.RELATIONSHIP_CONSTRAINTS, GENERIC_VARIABLES.OUTPUT_FORMAT,
                TDC_VARIABLES.SCHEMA_DETAILS
            ],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        self.prompt_context = {
            "column_names": column_names,
            "field_description": field_description,
            "max_length": max_length,
            "domain": domain,
            "feature_types": feature_types,
            "feature_distributions": feature_distributions,
            "relationships_constraints": relationships_constraints,
            "schema_details": schema_details
        }

    def initialize_metrics(self):
        """
        Initializes the prompt template and context for tabular data generation.
        This method prepares the reusable prompt template that will be used to generate LLM prompts.
        It sets up the structure of the prompt using the input variables, format instructions,
        and user-provided schema context.
        Args:
            column_names (str): Names of the columns in the tabular data.
            field_description (str): Description for each column/field.
            max_length (int): Maximum allowed character length per field.
            domain (str): Domain context of the dataset (e.g., retail, finance).
            feature_types (str): Types of features (e.g., numerical, categorical).
            feature_distributions (str): Statistical distributions expected in the features.
            relationships_constraints (str): Logical or statistical relationships among features.
            schema_details (dict): Full schema metadata including constraints and descriptions.
        Returns:
            None
        """
        return {
            "total_generated": 0,
            "duplicate_count": 0,
            "skipped_batches": 0,
            "cumulative_retries": 0,
            "batch_counts": 0,
        }
    
    def build_prompt(self, user_query, batch_size, realism_level, column_names,field_description, max_length, 
                      domain, feature_types,feature_distributions, relationships_constraints, output_format,
                      schema_details):
        """
        Builds a formatted prompt string for the synthetic data generation request.
        This method constructs a prompt by formatting a predefined template with the provided 
        parameters, which are used to generate synthetic tabular data. The prompt is tailored 
        based on user input, schema details, and various configuration options.

        Parameters:
            user_query (str): The query provided by the user to guide the data generation process.
            batch_size (int): The number of samples to generate in this batch.
            realism_level (str): The desired realism level of the generated data.
            column_names (str): A list of column names to include in the generated data.
            field_description (str): Descriptions of the fields to provide context for the generated data.
            max_length (int): The maximum length for text fields in the generated data.
            domain (str): The domain or context in which the data should fit (e.g., "finance").
            feature_types (str): The types of features to generate (e.g., "integer", "string").
            feature_distributions (str): The distribution types for the generated features (e.g., "normal", "uniform").
            relationships_constraints (str): Constraints related to the relationships between fields (e.g., "parent-child").
            output_format (str): The desired output format for the generated data (e.g., "JSON", "CSV").
            schema_details (dict): A dictionary containing details about the schema, such as column specifications.
        Returns:
            str: The formatted prompt string ready to be sent to the data generation model.
        Logs the token usage for the generated prompt using the `TokenEstimator` class to ensure the token count is tracked.
        """
            
        prompt = self.prompt_template.format(
                            user_request=user_query,
                            num_samples=batch_size,
                            realism_level=realism_level.strip(),
                            column_names=column_names,
                            field_description=field_description,
                            max_length=max_length,
                            domain=domain,
                            feature_types=feature_types,
                            feature_distributions=feature_distributions,
                            relationships_constraints=relationships_constraints,
                            output_format=output_format,
                            schema_details=json.dumps(schema_details)
                        )
        self.token_estimator.log_token_usage(prompt, label="Prompt")
        return prompt
    
    def generate_batch(self, prompt, seen_rows, batch_index, batch_size):
        """
        Generates a batch of synthetic tabular records using the language model and handles retries and deduplication.

        This method sends a formatted prompt to the language model to generate synthetic data. It includes:
        - Retry logic with exponential backoff for robustness.
        - Token usage logging for the model response.
        - Deduplication to ensure only unique records are kept.
        - Metric tracking for monitoring generation quality.

        Parameters:
            prompt (str): The prompt string to be sent to the language model.
            seen_rows (set): A set of previously seen records used to filter out duplicates.
            batch_index (int): The starting index of the batch within the total sample set.
            batch_size (int): The number of samples to generate in the batch.

        Returns:
            tuple:
                - List[dict]: Unique synthetic records generated in this batch.
                - dict: Batch-level metrics including:
                    - "new_records" (int): Number of unique records generated.
                    - "duplicates" (int): Number of duplicates identified in the batch.
                    - "retries" (int): Number of retry attempts made during generation.
                    - "skipped" (bool): Whether the batch was skipped due to persistent failures.

        Notes:
            - The method retries up to MAX_RETRIES times on failure or if only duplicate records are generated.
            - Token usage for the LLM response is logged using `TokenEstimator`.
            - Duplicate detection uses a set-based mechanism comparing record hashes or stringified values.
        """
        retry_count = 0
        batch_metrics = {
            "new_records": 0,
            "duplicates": 0,
            "retries": 0,
            "skipped": False
        }

        while retry_count < MAX_RETRIES:
            try:
                start_time = time.time()
                response = self.exponential_backoff_retry(lambda: self.llm_model.invoke([HumanMessage(content=prompt)]))
                if isinstance(response, tuple): 
                    response = response[0]
                self.token_estimator.log_token_usage(response.content, label="LLM Response")

                parsed_data = self.extract_json_from_response(response.content)
                unique_records, duplicate_count = self.deduplicate(parsed_data, seen_rows)

                if unique_records:
                    batch_metrics["new_records"] = len(unique_records)
                    batch_metrics["duplicates"] = duplicate_count
                    return unique_records, batch_metrics
                else:
                    self.logger.warning(f"No unique records in batch {batch_index // batch_size + 1}")
                    retry_count += 1
                    batch_metrics["retries"] += 1
            except Exception as e:
                self.logger.error(f"Error generating batch {batch_index // batch_size + 1}: {e}")
                retry_count += 1
                batch_metrics["retries"] += 1

        batch_metrics["skipped"] = True
        return [], batch_metrics
    
    def deduplicate(self, parsed_data, seen_rows):
        """
        Removes duplicate records from parsed data based on hash comparison.

        Parameters:
            parsed_data (list): List of generated records (typically dictionaries).
            seen_rows (set): Set of hashes representing previously seen records.

        Returns:
            tuple:
                - List[dict]: Unique records from the batch.
                - int: Number of duplicate records found and skipped.
        """
        unique_records = []
        duplicate_count = 0
        for record in parsed_data:
            row_hash = sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            if row_hash not in seen_rows:
                seen_rows.add(row_hash)
                unique_records.append(record)
            else:
                duplicate_count += 1
        return unique_records, duplicate_count
    
    def update_metrics(self, metrics, batch_metrics):
        """
        Updates the global generation metrics with the results from a single batch.

        Parameters:
            metrics (dict): The cumulative metrics being tracked for the full generation.
            batch_metrics (dict): The metrics returned from the latest batch generation.
        """
        metrics["total_generated"] += batch_metrics["new_records"]
        metrics["duplicate_count"] += batch_metrics["duplicates"]
        metrics["cumulative_retries"] += batch_metrics["retries"]
        if batch_metrics["new_records"] > 0:
            metrics["batch_counts"] += 1
        if batch_metrics["skipped"]:
            metrics["skipped_batches"] += 1

    def finalize_generation(self, all_data, metrics):
        """
        Removes duplicate records from parsed data based on hash comparison.

        Parameters:
            parsed_data (list): List of generated records (typically dictionaries).
            seen_rows (set): Set of hashes representing previously seen records.

        Returns:
            tuple:
                - List[dict]: Unique records from the batch.
                - int: Number of duplicate records found and skipped.
        """
        if not all_data:
            self.logger.error("No valid data generated.")
            return None

        df = pd.DataFrame(all_data)
        avg_per_batch = metrics["total_generated"] / metrics["batch_counts"] if metrics["batch_counts"] else 0

        self.logger.info(f"Final DataFrame with {len(df)} records.")
        self.logger.info(f"Total {metrics['total_generated']} unique records generated.")
        self.logger.info(f"Total {metrics['duplicate_count']} duplicate records skipped.")
        self.logger.info(f"{metrics['skipped_batches']} batches failed.")
        self.logger.info(f"Average records per batch: {avg_per_batch:.2f}")
        self.logger.info(f"Cumulative retry count: {metrics['cumulative_retries']}")

        return df,metrics["total_generated"]
    
    def save_generated_data(self, df, base_filename: str,output_format:str):
        """
        Save the generated tabular data in multiple formats using the base class logic.

        Args:
            df (pd.DataFrame): DataFrame containing the generated synthetic data.
            base_filename (str): Base name for the output file.
            output_format (str): Desired output format (e.g., CSV, JSON, Parquet).
        """

        if df is not None and (not hasattr(df, "empty") or not df.empty):
            self.save_output(df, base_filename,output_format)
            self.logger.info(f"Synthetic Tabular data saved successfully")
        else:
            self.logger.error("No data to save.")
