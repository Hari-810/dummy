import os
import logging
import tiktoken
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from config.constants import AZURE_MODELS_CONFIG

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelInitialization:
    def __init__(self):
        """
        Initializes the ModelInitialization class and loads environment variables.

        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing the Azure OpenAI model.")
        self.azure_endpoint = os.getenv(AZURE_MODELS_CONFIG.AZURE_OPENAI_ENDPOINT)
        self.api_version = os.getenv(AZURE_MODELS_CONFIG.API_VERSION)
        self.deployment_name = os.getenv(AZURE_MODELS_CONFIG.DEPLOYMENT_NAME)

        if not all([self.azure_endpoint, self.api_version, self.deployment_name]):
            raise ValueError("Missing required environment variables for Azure OpenAI configuration.")
        self.logger.info("Azure OpenAI configuration loaded successfully.")

    def get_base_model_name(self):
            """
              Determine  the base model for tokenizer from deployment name of the model.
            """
            if "gpt-4o-mini-v2024-07-18-ptu" in self.deployment_name:
                return "gpt-4o-mini"
            elif "gpt-4o-v2024-05-13-ptu" in self.deployment_name:
                return "gpt-4o"
            else:
                raise ValueError(f"Cannot determine base model for tokenizer from deployment name: {self.deployment_name}")
            
    def get_bearer_token(self):
        """
        Retrieves an authentication token for Azure Cognitive Services using DefaultAzureCredential.
        Logs success or failure based on the outcome.
        """
        try:
            token_provider = get_bearer_token_provider(DefaultAzureCredential(),AZURE_MODELS_CONFIG.BEARER_TOKEN_ENDPOINT)
            self.logger.info("Token retrieved successfully.")
            return token_provider
        except Exception as e:
            self.logger.error(f"Error fetching bearer token: {e}")
            return None

    def get_llm_model(self):
        """
        Initializes and returns the Azure OpenAI LLM model.
        """
        token = self.get_bearer_token()
        if token is None:
            raise RuntimeError("Failed to obtain Azure AD token for OpenAI LLM.")

        try:
            llm_model = AzureChatOpenAI(
                azure_endpoint=self.azure_endpoint,
                openai_api_version=self.api_version,
                deployment_name=self.deployment_name,
                openai_api_type=AZURE_MODELS_CONFIG.API_TYPE,
                azure_ad_token_provider=token,
                temperature=AZURE_MODELS_CONFIG.TEMPERATURE,
            )
            self.logger.info("Azure OpenAI LLM model initialized successfully.")
            return llm_model
        except Exception as e:
            self.logger.error(f"Error initializing LLM model: {e}")
            return None

    def get_tokenizer(self):
        """
        Returns the correct tokenizer based on the Azure OpenAI deployment.
        """
        base_model_name = self.get_base_model_name()
        tokenizer = tiktoken.encoding_for_model(base_model_name)  
        self.logger.info(f"Tokenizer initialized for model: {base_model_name}")
        return tokenizer
    