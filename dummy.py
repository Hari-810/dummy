import os
import logging
from typing import List, Dict, Union, Any
from azure.identity import DefaultAzureCredential
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from openai import AzureOpenAI
from openai._azure import get_bearer_token_provider

class AzureLanguageModel:
    """
    Azure OpenAI language model using Azure AD authentication, initialized from environment variables.
    Supports chat completions and embedding generation via LangChain.
    """

    def __init__(self, cache: bool = False):
        """Initialize Azure OpenAI configuration using environment variables and Azure AD token."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cache = cache
        self.response_cache: Dict[str, Any] = {} if self.cache else None

        # Load configuration from environment
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("API_VERSION")
        self.deployment_name = os.getenv("DEPLOYMENT_NAME")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL")

        if not all([self.azure_endpoint, self.api_version, self.deployment_name, self.embedding_model_name]):
            raise ValueError("Missing required Azure OpenAI environment variables.")

        self.logger.info("Azure OpenAI configuration loaded successfully.")

        # Initialize Azure AD token
        self.token_provider = self.get_bearer_token()
        if not self.token_provider:
            raise RuntimeError("Failed to obtain Azure AD token.")

        # Initialize LLM client
        self.llm_client = self.get_llm_model()

        # Token accounting
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost = 0.0

    def get_bearer_token(self):
        """Fetch Azure AD bearer token for Cognitive Services."""
        try:
            return get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default"
            )
        except Exception as e:
            self.logger.error(f"Error fetching bearer token: {e}")
            return None

    def get_llm_model(self):
        """Initialize the AzureChatOpenAI model with Azure AD token."""
        try:
            return AzureChatOpenAI(
                azure_endpoint=self.azure_endpoint,
                openai_api_version=self.api_version,
                deployment_name=self.deployment_name,
                openai_api_type="azure",
                azure_ad_token_provider=self.token_provider,
                max_tokens=4096,
                temperature=0.5,
            )
        except Exception as e:
            self.logger.error(f"Error initializing LLM model: {e}")
            return None

    def get_embedding_model(self):
        """Initialize the AzureOpenAIEmbeddings model."""
        try:
            return AzureOpenAIEmbeddings(
                model=self.embedding_model_name,
                azure_endpoint=self.azure_endpoint,
                openai_api_version=self.api_version,
                azure_ad_token_provider=self.token_provider,
            )
        except Exception as e:
            self.logger.error(f"Error initializing embedding model: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear the response cache."""
        if self.cache:
            self.response_cache.clear()
            self.logger.debug("Response cache cleared.")

    def query(self, query: str, num_responses: int = 1) -> Dict[str, Any]:
        """
        Query the Azure OpenAI chat model.
        :param query: The input prompt.
        :param num_responses: Number of responses to return.
        :return: Response object from the LLM.
        """
        if self.cache and query in self.response_cache:
            self.logger.debug(f"Cache hit for query: {query}")
            return self.response_cache[query]

        try:
            response = self.llm_client.generate([query], n=num_responses)
            self.logger.debug(f"Response generated with {len(response.generations)} choice(s).")

            if self.cache:
                self.response_cache[query] = response

            return response
        except Exception as e:
            self.logger.error(f"Error querying LLM: {e}")
            raise RuntimeError(f"Query failed: {e}")

    def get_response_texts(self, query_responses: Any) -> List[str]:
        """
        Extract response texts from the AzureChatOpenAI response object.
        :param query_responses: LangChain LLMResult or similar response object.
        :return: List of response texts.
        """
        try:
            return [gen.text for gen in query_responses.generations[0]]
        except Exception as e:
            self.logger.error(f"Error extracting response texts: {e}")
            return []
