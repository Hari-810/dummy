import os
import yaml
import pandas as pd
import logging
import time
from datetime import datetime
from abc import ABC, abstractmethod
from config.model_config import ModelInitialization
from config.constants import PROMPT_DIR, TABULAR_DATA_FOLDER, TEXTUAL_DATA_FOLDER,RETRY_CONFIG

# Configure module-level logger
logger = logging.getLogger(__name__)

class BaseSyntheticGenerator(ABC):
    """
    Abstract base class for synthetic data generation using an LLM model.
    Loads prompt templates, interfaces with the model, and supports data saving and retry logic.
    """

    def __init__(self, data_category):
        """
        Initialize the generator with a specific data category and LLM model.

        Args:
            data_category (str): The type of data to generate (e.g., "tabular", "textual").
        """
        self.data_category = data_category.lower()
        logger.info("Initializing Azure Open AI Model for %s data", self.data_category)
        self.model_init = ModelInitialization()
        self.llm_model = self.model_init.get_llm_model()
        self.tokenizer = self.model_init.get_tokenizer()
        self.prompt_template = self.load_prompt_template()
        

    def load_prompt_template(self):
        """
        Load YAML prompt template dynamically based on the data category.

        Returns:
            dict: Parsed YAML data for prompts.

        Raises:
            FileNotFoundError: If the YAML file is not found.
            ValueError: If the YAML content is empty or invalid.
        """
        yaml_path = os.path.join(PROMPT_DIR, f"{self.data_category}_data.yaml")
        logger.debug("Loading prompt template from: %s", yaml_path)

        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Prompt file for {self.data_category}_data not found!")

        with open(yaml_path, "r", encoding="utf-8") as file:
            prompt_data = yaml.safe_load(file)

        if not prompt_data:
            raise ValueError(f"YAML file for {self.data_category} is empty or invalid.")

        return prompt_data
    
    def handle_backoff_sleep(self, delay: float) -> float:
        """
        Sleep for the given delay and log the duration.
        Args:
            delay (float): Number of seconds to sleep.
        Returns:
            float: The sleep time actually waited.
        """
        logger.info(f"Sleeping for {delay} seconds before retrying...")
        time.sleep(delay)
        return delay
    
    def exponential_backoff_retry(self, func,*args,**kwargs):
        """
        A helper function that wraps a call to another function and retries on failure with exponential backoff.
        Args:
            func (callable): The function to call and retry.
            max_retries (int): Maximum number of retry attempts.
            initial_delay (int): Initial delay before retrying (in seconds).
            backoff_factor (int): The factor by which the delay increases after each retry.

        Returns:
            The result of the function call if successful, otherwise raises the last exception.
        """
        attempt = 0
        delay = RETRY_CONFIG["initial_delay"]
        total_sleep = 0

        while attempt < RETRY_CONFIG["max_retries"]:
            try:
                result = func(*args, **kwargs)
                return result,attempt,total_sleep
            except Exception as e:
                attempt += 1
                logger.error(f"Attempt {attempt} failed: {e}")
                if attempt < RETRY_CONFIG["max_retries"]:
                    slept = self.handle_backoff_sleep(delay)
                    total_sleep += slept
                    logger.info(f"Retrying in {delay} seconds...")
                    delay *= RETRY_CONFIG["backoff_factor"] 
                else:
                    logger.error("Maximum retries reached. Raising the last exception.")
                    raise e

    @abstractmethod
    def generate_synthetic_data(self, user_query: str, num_samples: int):
        """
        Generate synthetic data using the LLM model. Must be implemented by subclasses.

        Args:
            user_query (str): The user-defined instruction or input.
            num_samples (int): Number of data samples to generate.

        Returns:
            Generated data (varies by implementation).
        """
        pass

    def save_output(self, data, output_file: str, output_format: str):
        """
        Save generated synthetic data to a file, based on its category and format.

        Args:
            data: The generated data (DataFrame or list).
            output_file (str): Base filename for output.
            output_format (str): Output file format (CSV, JSON, Parquet, or TXT).
        """
        if isinstance(data, pd.DataFrame) and data.empty:
            logger.info("No data to save (DataFrame is empty).")
            return ""
        if isinstance(data, list) and not data:
            logger.info("No data to save (empty list).")
            return ""
        if data is None:
            logger.info("No data to save (None).")
            return ""
        
        output_format = output_format.lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_method = getattr(self, f"_save_{self.data_category.replace(' ', '_')}", None)

        if callable(save_method):
            return save_method(data, output_file, timestamp, output_format)
        else:
            logger.error("Unsupported save operation for category: %s", self.data_category)
            return ""

    def _save_tabular(self, data, base_filename: str, timestamp: str, output_format: str):
        """
        Save tabular data in CSV, JSON, or Parquet format.

        Args:
            data (pd.DataFrame): Generated tabular data.
            base_filename (str): Base name for the file.
            timestamp (str): Timestamp string to append.
            output_format (str): Desired file format.
        """
        df = pd.DataFrame(data)
        base_path = os.path.join(TABULAR_DATA_FOLDER, f"{base_filename}_{timestamp}")

        if output_format.upper() == "CSV":
            save_path = f"{base_path}.csv"
            df.to_csv(save_path , index=False)

        elif output_format.upper() == "JSON":
            save_path = f"{base_path}.json"
            df.to_json(save_path, orient="records", indent=2)

        elif output_format.lower() == "parquet":
            save_path = f"{base_path}.parquet"
            df.to_parquet(save_path, index=False)

        logger.info("Synthetic tabular data saved to: %s", save_path)
        return save_path

    def _save_textual(self, data, base_filename: str, timestamp: str, output_format: str):
        """
        Save textual data to a plain text file.

        Args:
            data (list): Generated natural text data.
            base_filename (str): Base name for the file.
            timestamp (str): Timestamp string to append.
            output_format (str): Desired file format (should be 'TXT').
        """
        save_path = os.path.join(TEXTUAL_DATA_FOLDER, f"{base_filename}_{timestamp}.txt")
        with open(save_path, "w", encoding="utf-8") as file:
            file.write("\n\n".join(data))

        logger.info("Synthetic natural text data saved to: %s", save_path)
        return save_path