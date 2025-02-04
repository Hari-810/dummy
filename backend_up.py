

PLUGIN_CREATION_PROMPT_DESC= """
sample_python_file: {
"""
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
sample_yaml_file: {"""
    name: #plugin name
    enabled: true
    required: false
    custom_modification: #based on the custom input provided. (it can be trueor false. by default set it to false.) 

    # Instruction - Add the configuration segment and set the values based on the input passed. if the deployment_name or endpoint_url is None or empty just initialize all the variables in the configuration to "None"
    configurations:
      api_type: # azure_ad
      deployment_name: # gpt-4o-mini-v2024-07-18-ptu
      endpoint_url: # https://aimlameuse2npdopenai.openai.azure.com/
      frequency_penalty: # 0
      openai_version: # 2024-02-15-preview
      presence_penalty: #0
      temperature: #0.1
      top_p: #0.95


    description: >-
      # instruction - add detailed description on what this plugin does.
      ** Below line should be in every plugin creation
      "**Should use only {plugin_name=current name of the plugin} function for the response"
      The description should have the following details:
      1. What this plugin does.
      2. what is the input and its type that should be passed.
      3. What are the does and donts that should be considered while processing the inputs.
      4. what is the required output and its type from the plugin.

    parameters:
      - name: #input parameter name
        type: #input parameter type
        required: true
        description: #description on the input parameter passed
      - name: #input parameter name
        type: #input parameter type
        required: true
        description: #description on the input parameter passed

    #Instruction - change the return parameters based on the requirements. the return type may be a string, int, file or any other format requested.
    returns:
      - name: #output parameter name
        type: #output parameter type
        description: #description on the output parameter passed.

      - name: #output parameter name
        type: #output parameter type
        description: #description on the output parameter passed.
    """
}

plugin_name = {plugin_name}
plugin_description = {plugin_description}
mandatory_conditions =  {mandatory_conditions}
output_type = {output_type}
input_parameters = {input_parameters}



**Should use only dyanmic_plugin_creation function for the response
Based on the provided inputs, create .py and .yaml file for the new plugin by following the sample templates provided.
  - python file created must follow given sample_python_file template.
  - yaml file created must follow given  sample_yaml_file template
  - The name for the .py and .yaml should be the same as plugin_name provided.
  - Generat all necessary code logics to support the description
  - Import all necessary libraries for the plugins to work.
  - Implement exception handling on the .py file generated.

* NOTE: Instructions to be followed while generating the python and yaml files for new plugin which is going to be created:
  - **Should use only dyanmic_plugin_creation function for the response
  - The .py and .yaml file should strictly follow the sample template pattern. 
  - The generated files should have all the necessary syntax to fit in as plugin inside Taskweaver framework.
  - Strictly follow the sample template format for generating .py and .yaml files. if any of the values found missing, just replace it with default values from the sample template.
  - Follow the instructions mentioned in the sample template for generating the contents but do not copy the instruction to the new files generated.
  - Strictly do not generate files structure on your own in diffrent format. only pattern instructed in sample files should be followed.

* IMPORTANT Please generate the .py and .yaml files for the given requirements. provide the output in Dictionary format as shown below.
  - Example output format:
      "python_file": contents of the .py file generated,
      "yaml_file": contents of the .yaml file generated,

* Self-assessment:
  - Analyze the .py file generated and compare it with sample_python_file. Highlight the structural differences and adjust the ouput to match the sample excatly. 
  - Analyze the .yaml file generated and compare it with sample_yaml_file. Highlight the structural differences and adjust the ouput to match the sample excatly.
  - if the ouput dosent match the samples, correct it and regenerate unitil it adheres to the format.
  - validate the python and yaml file generated with the checklist = {sefl_assessment_checklist}
  ** Do not add any instructions or any comments in the newly created python and yaml files.**
"""

