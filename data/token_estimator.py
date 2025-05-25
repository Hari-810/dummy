from config.model_config import ModelInitialization
from config.constants import MODEL_INPUT_LIMIT,MODEL_OUTPUT_LIMIT
import logging

# Configure module-level logger
logger = logging.getLogger(__name__)

class TokenEstimator:
    def __init__(self):
        self.model_init = ModelInitialization()
        self.tokenizer = self.model_init.get_tokenizer()

    def log_token_usage(self, text: str, label: str = "Input"):
        """
        Logs the number of tokens used in a given text input.

        Parameters:
            text (str): The text whose token count should be calculated.
            label (str): A label to describe the source of the token count 
                        (e.g., "Input", "Prompt", "LLM Response").

        Returns:
            None
        """
        token_count = len(self.tokenizer.encode(text))
        logger.info(f"{label} token count: {token_count}")
        return token_count


    def estimate_tokens_for_schema(self, schema_details):
        """
        Estimates the number of tokens based on schema details, using the tokenizer for more accuracy.

        Args:
        schema_details (dict): Schema details provided in the payload.
        
        Returns:
        int: Estimated token usage based on column names, descriptions, etc.
        """
        total_tokens = 0
        for column_name, metadata in schema_details.items():
            column_name_tokens = len(self.tokenizer.encode(column_name)) 
            description_tokens = len(self.tokenizer.encode(metadata.get('field_description', '')))  
            max_length_tokens = len(self.tokenizer.encode(str(metadata.get('max_length', ''))))  
            total_tokens += column_name_tokens + description_tokens + max_length_tokens
        return total_tokens
    
    def estimate_output_tokens(self, schema_tokens: int, prompt_tokens: int) -> int:
        """
        Dynamically estimates the number of output tokens based on schema and prompt tokens.
        
        Args:
        schema_tokens (int): Estimated tokens for the schema.
        prompt_tokens (int): Estimated tokens for the input prompt.
        
        Returns:
        int: Estimated output tokens per record.
        """
        return max(20, schema_tokens // 10)  
    
    def get_dynamic_batch_size(self, schema_tokens: int, prompt_tokens: int) -> int:
        """
        Dynamically calculates the batch size based on schema tokens, prompt tokens, model input/output limits.
        
        :param schema_tokens: Estimated tokens based on schema.
        :param prompt_tokens: Estimated tokens for the input prompt.
        :return: Calculated batch size.
        """
        total_input_tokens = schema_tokens + prompt_tokens
        total_output_tokens_per_record = self.estimate_output_tokens(schema_tokens, prompt_tokens)
        
        max_input_records = (MODEL_INPUT_LIMIT - total_input_tokens) // total_input_tokens     
        max_output_records = MODEL_OUTPUT_LIMIT // total_output_tokens_per_record
        
        dynamic_batch_size = min(max_input_records, max_output_records)
        
        return max(1, dynamic_batch_size)