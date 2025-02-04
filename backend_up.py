PLUGIN_CREATION_PROMPT_DESC = """
sample_python_file: {
from taskweaver.plugin import Plugin, register_plugin
from langchain_openai import AzureChatOpenAI
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import os
import requests

load_dotenv()

@register_plugin
class SamplePlugin(Plugin):  # Replaced {PluginClassName} with SamplePlugin
    # Task Name: SampleTask
    # Description: Sample plugin for API interaction

    def __call__(self, inputs: dict):
        \"\"\"
        Instructions:
        1. **Input Validation**:
            - Ensure that the `inputs` dictionary contains all required parameters.
            - The required parameters could include things like `api_endpoint` (the API's URL) and `query_parameters` (the parameters for the API request).
            - If any required parameters are missing, raise an exception (e.g., `ValueError`) to indicate which parameter is missing.
        2. **API Interaction and Token Management**:
            - Retrieve the necessary parameters from the `inputs` dictionary.
            - For example, extract `api_endpoint` and `query_parameters`.
            - If token-based authentication is required (e.g., using Azure):
                - Use `DefaultAzureCredential` or similar utilities to retrieve a token.
                - Include the token in the API request header, typically as a Bearer token.
        3. **Make the API Request**:
            - Use the `api_endpoint` and `query_parameters` to initiate the API call.
            - For example, you may use the `requests.get()` function to call the API with the appropriate parameters and headers.
            - Ensure that the headers contain the necessary authentication token if required.
        4. **Response Processing**:
            - After the API call, process the response data.
            - Extract necessary fields from the response, such as `status`, `data`, or any other relevant information based on your specific use case.
            - Optionally, transform or filter the response data to match the desired output format.
        5. **Return Processed Data**:
            - Return the processed data in a dictionary format.
            - The dictionary should contain the relevant data, such as:
                - `status`: The result of the task (e.g., success or failure).
                - `data`: The processed data extracted from the API response.
        \"\"\"
        if 'api_endpoint' not in inputs:
            raise ValueError("Missing required parameter: api_endpoint")
        
        api_endpoint = inputs["api_endpoint"]
        query_parameters = inputs.get("query_parameters", {})

        # Token-based authentication (if required)
        credential = DefaultAzureCredential()
        token = credential.get_token("https://management.azure.com/.default").token

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        response = requests.get(api_endpoint, params=query_parameters, headers=headers)

        return {
            "status": response.status_code,
            "data": response.json() if response.ok else response.text
        }
"""

sample_yaml_file = """
    name: plugin_name
    enabled: true
    required: false
    custom_modification: false

    configurations:
      api_type: azure_ad
      deployment_name: gpt-4o-mini-v2024-07-18-ptu
      endpoint_url: https://aimlameuse2npdopenai.openai.azure.com/
      frequency_penalty: 0
      openai_version: 2024-02-15-preview
      presence_penalty: 0
      temperature: 0.1
      top_p: 0.95

    description: >-
      The description should have the following details:
      1. What this plugin does.
      2. what is the input and its type that should be passed.
      3. What are the does and donts that should be considered while processing the inputs.
      4. what is the required output and its type from the plugin.

    parameters:
      - name: input_parameter_name
        type: input_parameter_type
        required: true
        description: description of the input parameter
      - name: another_input_parameter
        type: another_input_parameter_type
        required: true
        description: description of the input parameter

    returns:
      - name: output_parameter_name
        type: output_parameter_type
        description: description of the output parameter
      - name: another_output_parameter
        type: another_output_parameter_type
        description: description of the output parameter
"""

plugin_name = "plugin_name"
plugin_description = "plugin_description"
mandatory_conditions = "mandatory_conditions"
output_type = "output_type"
input_parameters = "input_parameters"

# Note: Do not include instructions or comments in the newly created files.
