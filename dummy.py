# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any
import json
import os
import logging


class AbstractLanguageModel(ABC):
    """
    Abstract base class that defines the interface for all language models.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "", cache: bool = False
    ) -> None:
        """
        Initialize the AbstractLanguageModel instance with configuration, model details, and caching options.

        :param config_path: Path to the config file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the language model. Defaults to "".
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config: Dict = None
        self.model_name: str = model_name
        self.cache = cache
        if self.cache:
            self.respone_cache: Dict[str, List[Any]] = {}
        self.load_config(config_path)
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.cost: float = 0.0

    def load_config(self, path: str) -> None:
        """
        Load configuration from a specified path.

        :param path: Path to the config file. If an empty path provided,
                     default is `config.json` in the current directory.
        :type path: str
        """
        if path == "":
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, "config.json")

        with open(path, "r") as f:
            self.config = json.load(f)

        self.logger.debug(f"Loaded config from {path} for {self.model_name}")

    def clear_cache(self) -> None:
        """
        Clear the response cache.
        """
        self.respone_cache.clear()

    @abstractmethod
    def query(self, query: str, num_responses: int = 1) -> Any:
        """
        Abstract method to query the language model.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: The number of desired responses.
        :type num_responses: int
        :return: The language model's response(s).
        :rtype: Any
        """
        pass

    @abstractmethod
    def get_response_texts(self, query_responses: Union[List[Dict], Dict]) -> List[str]:
        """
        Abstract method to extract response texts from the language model's response(s).

        :param query_responses: The responses returned from the language model.
        :type query_responses: Union[List[Dict], Dict]
        :return: List of textual responses.
        :rtype: List[str]
        """
        pass

import os
import json
import logging
from typing import List, Dict, Union, Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from abstract_language_model import AbstractLanguageModel  # adjust import as needed


class AzureADLanguageModel(AbstractLanguageModel):
    def __init__(self, config_path: str = "", model_name: str = "", cache: bool = False):
        super().__init__(config_path, model_name, cache)
        self._load_env_or_config()
        self.credential = DefaultAzureCredential()
        self.token_provider = self._get_bearer_token_provider()
        self.llm_model = self._init_llm()
        self.embedding_model = self._init_embedding()

    def _load_env_or_config(self):
        """
        Loads config values from environment variables or config file.
        """
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", self.config.get("AZURE_OPENAI_ENDPOINT"))
        self.api_version = os.getenv("API_VERSION", self.config.get("AZURE_OPENAI_API_VERSION"))
        self.deployment_name = os.getenv("DEPLOYMENT_NAME", self.config.get("AZURE_OPENAI_DEPLOYMENT"))
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", self.config.get("EMBEDDING_MODEL"))

        if not all([self.azure_endpoint, self.api_version, self.deployment_name, self.embedding_model_name]):
            raise ValueError("Missing Azure OpenAI configuration.")

    def _get_bearer_token_provider(self):
        """
        Returns a callable that provides a bearer token.
        """
        try:
            return get_bearer_token_provider(
                self.credential,
                "https://cognitiveservices.azure.com/.default"
            )
        except Exception as e:
            logging.error(f"Error creating bearer token provider: {e}")
            raise

    def _init_llm(self):
        """
        Initializes the AzureChatOpenAI model using Azure AD token provider.
        """
        try:
            return AzureChatOpenAI(
                azure_endpoint=self.azure_endpoint,
                openai_api_version=self.api_version,
                deployment_name=self.deployment_name,
                azure_ad_token_provider=self.token_provider,
                openai_api_type="azure",
                max_tokens=4096,
                temperature=0.5,
            )
        except Exception as e:
            logging.error(f"Failed to initialize AzureChatOpenAI: {e}")
            return None

    def _init_embedding(self):
        """
        Initializes the Azure OpenAI Embedding model using Azure AD.
        """
        try:
            return AzureOpenAIEmbeddings(
                model=self.embedding_model_name,
                azure_endpoint=self.azure_endpoint,
                openai_api_version=self.api_version,
                azure_ad_token_provider=self.token_provider,
            )
        except Exception as e:
            logging.error(f"Failed to initialize AzureOpenAIEmbeddings: {e}")
            return None

    def query(self, query: str, num_responses: int = 1) -> Any:
        """
        Query the AzureChatOpenAI model.
        """
        try:
            if not self.llm_model:
                raise RuntimeError("LLM model not initialized.")
            messages = [{"role": "user", "content": query}]
            response = self.llm_model.invoke(messages)
            if self.cache:
                self.respone_cache[query] = response
            return response
        except Exception as e:
            logging.error(f"Error during LLM query: {e}")
            return None

    def get_response_texts(self, query_responses: Union[List[Dict], Dict]) -> List[str]:
        """
        Extract text content from the model response.
        """
        try:
            # Single response object (LangChain `AIMessage`)
            if isinstance(query_responses, dict) and "content" in query_responses:
                return [query_responses["content"]]
            elif hasattr(query_responses, "content"):
                return [query_responses.content]
            # List of response dicts
            elif isinstance(query_responses, list):
                return [resp.get("message", {}).get("content", "") for resp in query_responses]
            else:
                raise TypeError("Invalid response format.")
        except Exception as e:
            logging.error(f"Failed to extract response texts: {e}")
            return []
        

model = AzureADLanguageModel()
response = model.query("What is GraphRAG?")
print(model.get_response_texts(response))
