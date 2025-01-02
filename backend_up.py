import atexit
import base64
import csv
import functools
import json
import logging as log
import mimetypes
import os
import re
import shutil
import sys
import ast
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib import request

import ast
import msal
import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from ruamel.yaml import YAML

from azure.core.credentials import AccessToken
from azure.identity import DefaultAzureCredential
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.response import Response
from rest_framework.renderers import BaseRenderer

from langchain_openai import AzureChatOpenAI

# Local imports
from .models import *
from .models import APIDetails, LLMModel
from .serializers import *
from .serializers import APIDetailsSerializer, BlobDataSerializer

# TaskWeaver imports
from taskweaver.app.app import TaskWeaverApp  # Import TaskWeaverApp
from taskweaver.code_interpreter import code_executor
from taskweaver.memory.attachment import AttachmentType
from taskweaver.memory.plugin import set_plugin_folder
from taskweaver.memory.type_vars import RoleName
from taskweaver.module.event_emitter import (
    PostEventType,
    RoundEventType,
    SessionEventHandler,
    SessionEventHandlerBase,
    TaskWeaverEvent,
)
from taskweaver.session.session import Session


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve repository path
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
PROJECT_PATH = os.path.join(REPO_PATH, "project")

# Append repo_path to sys.path for imports
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

# Initialize global variables
app_session_dict: Dict[str, Session] = {}
admin_session_dict: Dict[str, Session] = {}
draft_session_dict: Dict[str, Session] = {}

def initialize_app_sessions():
    """
    Initializes the TaskWeaverApp and session dictionaries for app, admin, and draft plugins.
    """
    try:
        global app_session_dict, admin_session_dict, draft_session_dict
        
        # Initialize main app
        app = TaskWeaverApp(app_dir=PROJECT_PATH, use_local_uri=True)
        atexit.register(app.stop)
        app_session_dict["default"] = app.get_session()

        # Initialize admin app
        admin_app = TaskWeaverApp(app_dir=PROJECT_PATH, use_local_uri=True)
        atexit.register(admin_app.stop)
        admin_session_dict["default"] = admin_app.get_session()

        # Initialize draft app
        draft_app = TaskWeaverApp(app_dir=PROJECT_PATH, use_local_uri=True)
        atexit.register(draft_app.stop)
        draft_session_dict["default"] = draft_app.get_session()

        # Set plugin folders
        code_executor.set_plugin_folder("adminplugins")
        set_plugin_folder("adminplugins")
        code_executor.set_plugin_folder("draftplugins")
        set_plugin_folder("draftplugins")

        logger.info("Applications and sessions initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing applications: {str(e)}")
        sys.exit(1)

# Initialize applications
initialize_app_sessions()

@api_view(['GET'])
def is_authorized(request):
    """
    Description:
        Fetches the username from the frontend and retrieves an access token using MSAL.
        This token is used to get the members of a specific group via Microsoft Graph API.
        It checks if the username matches any member's UserPrincipalName and returns the result.
    
    Parameters:
        request: The HTTP request object used to extract headers.
    
    Returns:
        JSON response in the format: {"Access": True/False}.
    """
    # Get the username from the request headers
    username = request.headers.get('username')
    if not username:
        return JsonResponse({"error": "Username header is missing"}, status=400)
    
    # MSAL configuration
    config = {
        'client_id': os.getenv("ClientId"),
        'client_secret': os.getenv("ClientSecret"),
        'authority': os.getenv("Instance") + os.getenv("TenantId"),
        'scope': ['https://graph.microsoft.com/.default']
    }
    
    client = msal.ConfidentialClientApplication(
        client_id=config['client_id'],
        authority=config['authority'],
        client_credential=config['client_secret']
    )
    
    # Acquire token
    token_result = client.acquire_token_silent(config['scope'], account=None)
    if not token_result:
        token_result = client.acquire_token_for_client(scopes=config['scope'])
        if 'access_token' not in token_result:
            return JsonResponse({"error": "Failed to acquire access token", "details": token_result}, status=500)
    
    # Fetch group members using Microsoft Graph API
    headers = {'Authorization': f"Bearer {token_result['access_token']}"}
    url = f"https://graph.microsoft.com/v1.0/groups/{os.getenv('GroupId_1')}/members"
    azure_group_members = []
    
    try:
        while url:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            azure_group_members.extend(data.get('value', []))
            url = data.get('@odata.nextLink')  # Pagination
    except requests.RequestException as e:
        return JsonResponse({"error": "Failed to fetch group members", "details": str(e)}, status=500)
    
    # Check if username matches any group member
    is_username_present = any(user.get("userPrincipalName") == username for user in azure_group_members)
    return JsonResponse({"Access": is_username_present}, safe=False)

@api_view(['GET'])
def list_plugins(request):
    """
    Retrieves a list of plugins from the plugins folder, including their details.

    The function reads `.yaml` files in the plugins folder to extract details
    such as `plugin_name`, `description`, `examples`, `parameters`, and `return values`.
    Updates a CSV file with these details if there are changes.

    Returns:
        JsonResponse: A list of dictionaries containing plugin details.
    """
    # Define file paths
    plugins_folder_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'project', 'plugins'))
    csv_file_path = os.path.join(os.getcwd(), 'plugins.csv')

    # Validate plugin folder path
    if not os.path.exists(plugins_folder_path):
        return JsonResponse({"error": "Plugins folder not found."}, status=404)

    # Get all .yaml files in the plugins folder
    yaml_files = [
        f for f in os.listdir(plugins_folder_path)
        if f.endswith('.yaml') and os.path.isfile(os.path.join(plugins_folder_path, f))
    ]

    # Load existing plugin data from CSV
    existing_plugins = {}
    if os.path.exists(csv_file_path):
        with open(csv_file_path, mode='r') as csv_file:
            reader = csv.DictReader(csv_file)
            existing_plugins = {row['plugin_name']: row for row in reader}

    plugins = []

    # Process each YAML file
    for yaml_file in yaml_files:
        file_path = os.path.join(plugins_folder_path, yaml_file)
        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
        except Exception as e:
            return JsonResponse({"error": f"Failed to parse {yaml_file}: {str(e)}"}, status=500)

        # Extract plugin details
        plugin_name = yaml_file[:-5]
        description = data.get('description', 'No description available')
        enabled_status = data.get('enabled', 'No enabled status available')
        example = data.get('examples', 'No example available')
        custom_modification = data.get('custom_modification', False)
        plugin_status = data.get('isdeleted', False)
        configurations = data.get("configurations", {})
        endpoint = configurations.get('endpoint_url', 'No Endpoint available')
        deployment_name = configurations.get('deployment_name', 'No deployment name available')

        # Extract parameter descriptions
        parameter_description = "; ".join(
            f"{param['name']} ({param['type']}): {param.get('description', 'No description')}"
            for param in data.get('parameters', [])
        )
        # Extract return descriptions
        return_description = "; ".join(
            f"{ret['name']} ({ret['type']}): {ret.get('description', 'No description')}"
            for ret in data.get('returns', [])
        )

        plugin_info = {
            "plugin_name": plugin_name,
            "description": description,
            "example": example,
            "parameter_description": parameter_description,
            "return_description": return_description,
            "enabled_status": enabled_status,
            "custom_modification": custom_modification,
            "Endpoint": endpoint,
            "DEPLOYMENT_NAME": deployment_name,
            "Is_deleted": plugin_status,
        }
        plugins.append(plugin_info)

        # Update existing plugin records
        existing_plugins[plugin_name] = plugin_info

    # Write updated data back to CSV
    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = [
            "plugin_name", "description", "example", "parameter_description",
            "return_description", "enabled_status", "custom_modification",
            "Endpoint", "DEPLOYMENT_NAME", "Is_deleted"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_plugins.values())

    return JsonResponse(plugins, safe=False)

@api_view(['GET'])
def list_adminplugins(request):
    """
    Retrieves a list of plugin names from the admin plugins folder.

    Returns:
        JsonResponse: A list of dictionaries containing plugin names.
        Each dictionary includes:
            - `plugin_name` (str): Name of the plugin file without extension.
    """
    # Define the admin plugins folder path
    plugins_folder_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'project', 'adminplugins'))

    # Validate folder existence
    if not os.path.exists(plugins_folder_path):
        return JsonResponse({"error": "Admin plugins folder not found."}, status=404)

    try:
        # List all YAML files in the folder
        yaml_files = [
            f for f in os.listdir(plugins_folder_path)
            if f.endswith('.yaml') and os.path.isfile(os.path.join(plugins_folder_path, f))
        ]
    except Exception as e:
        return JsonResponse({"error": f"Failed to access the folder: {str(e)}"}, status=500)

    # Extract plugin names
    plugins = [{"plugin_name": yaml_file[:-5]} for yaml_file in yaml_files]

    return JsonResponse(plugins, safe=False)

@api_view(['GET'])
def list_draftplugins(request):
    """
    Retrieves a list of plugins from the draft plugins folder, including their details.

    Returns:
        JsonResponse: A list of dictionaries containing plugin details:
            - `plugin_name` (str): Plugin file name without extension.
            - `description` (str): Plugin description or default message.
            - `example` (str): Example usage of the plugin.
            - `parameter_description` (str): Description of parameters.
            - `return_description` (str): Description of return values.
            - `enabled_status` (str): Enabled status.
    """
    # Define folder path
    plugins_folder_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'project', 'draftplugins'))

    # Validate folder existence
    if not os.path.exists(plugins_folder_path):
        return JsonResponse({"error": "Draft plugins folder not found."}, status=404)

    plugins = []
    try:
        # Get all YAML files in the folder
        yaml_files = [
            f for f in os.listdir(plugins_folder_path)
            if f.endswith('.yaml') and os.path.isfile(os.path.join(plugins_folder_path, f))
        ]

        # Process each YAML file
        for yaml_file in yaml_files:
            file_path = os.path.join(plugins_folder_path, yaml_file)
            try:
                with open(file_path, 'r') as file:
                    data = yaml.safe_load(file) or {}
            except Exception as e:
                return JsonResponse({"error": f"Failed to parse {yaml_file}: {str(e)}"}, status=500)

            # Extract plugin details
            plugin_name = yaml_file[:-5]
            description = data.get('description', 'No description available')
            enabled_status = data.get('enabled', 'No enabled status available')
            example = data.get('examples', 'No example available')
            custom_modification = data.get('custom_modification', False)
            plugin_status = data.get('isdeleted', False)
            configurations = data.get('configurations', {})
            endpoint = configurations.get('endpoint_url', 'No Endpoint available')
            deployment_name = configurations.get('deployment_name', 'No deployment name available')

            # Extract parameter and return descriptions
            parameter_description = "; ".join(
                f"{param['name']} ({param['type']}): {param.get('description', 'No description')}"
                for param in data.get('parameters', [])
            )
            return_description = "; ".join(
                f"{ret['name']} ({ret['type']}): {ret.get('description', 'No description')}"
                for ret in data.get('returns', [])
            )

            # Add plugin details to the list
            plugins.append({
                "plugin_name": plugin_name,
                "description": description,
                "example": example,
                "parameter_description": parameter_description,
                "return_description": return_description,
                "enabled_status": enabled_status,
                "custom_modification": custom_modification,
                "Endpoint": endpoint,
                "DEPLOYMENT_NAME": deployment_name,
                "Is_deleted": plugin_status,
            })

    except Exception as e:
        return JsonResponse({"error": f"An unexpected error occurred: {str(e)}"}, status=500)

    return JsonResponse(plugins, safe=False)


@api_view(['POST'])
def plugin_enable(request):
    """
    Enables or disables a plugin by modifying its YAML configuration.

    Args:
        request (HttpRequest): The POST request containing:
            - `plugin_name` (str): Name of the plugin to enable/disable.
            - `state` (bool): Desired state of the plugin (True for enable, False for disable).

    Returns:
        HttpResponse: Status message indicating success or failure.
    """
    # Extract data from the request
    data = request.data
    plugin_name = data.get('plugin_name')
    plugin_state = data.get('state')

    if not plugin_name:
        return HttpResponse(
            "Plugin name is required.",
            content_type="text/plain",
            status=status.HTTP_400_BAD_REQUEST
        )

    if not isinstance(plugin_state, bool):
        return HttpResponse(
            "State must be a boolean value.",
            content_type="text/plain",
            status=status.HTTP_400_BAD_REQUEST
        )

    # Define the path to the plugins folder
    plugins_folder = os.path.abspath(os.path.join(os.getcwd(), '..', 'project', 'plugins'))
    yaml_file = os.path.join(plugins_folder, plugin_name + ".yaml")

    # Ensure the file exists
    if not os.path.exists(yaml_file):
        return HttpResponse(
            f"YAML file for plugin '{plugin_name}' not found.",
            content_type="text/plain",
            status=status.HTTP_404_NOT_FOUND
        )

    try:
        # Load the YAML file with ruamel.yaml
        yaml = YAML()
        yaml.preserve_quotes = True  # Preserve original YAML formatting

        with open(yaml_file, 'r') as file:
            config = yaml.load(file)

        # Modify the 'enabled' property
        config['enabled'] = bool(plugin_state)

        # Save the updated YAML file
        with open(yaml_file, 'w') as file:
            yaml.dump(config, file)

        return HttpResponse(
            "YAML file updated successfully.",
            content_type="text/plain",
            status=status.HTTP_200_OK
        )
    except Exception as e:
        # Handle unexpected errors
        return HttpResponse(
            f"An error occurred while updating the YAML file: {str(e)}",
            content_type="text/plain",
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
@api_view(['GET'])
def list_llm_models(request):
    """
    Retrieves a list of LLM model details from the APIDetails model.

    Returns:
        JsonResponse: A JSON array of all entries from the APIDetails model.
    """
    try:
        # Fetch all entries from APIDetails as dictionaries
        model_names = APIDetails.objects.values()
        models_name_list = list(model_names)
        
        return JsonResponse(models_name_list, safe=False, status=status.HTTP_200_OK)
    except Exception as e:
        # Handle unexpected errors
        return JsonResponse(
            {"error": f"An error occurred: {str(e)}"},
            safe=False,
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
def llmconfig(request):
    """
    Updates the taskweaver_config.json file with LLM configuration details 
    based on the provided deployment model.

    Args:
        request: The HTTP request containing the deployment model in the body.

    Returns:
        HttpResponse: A plain text response indicating success or failure.
    """
    try:
        # Extract data from the request
        data = request.data
        deployment_model = data.get('deployment_model')

        if not deployment_model:
            return HttpResponse(
                "Deployment model is required.", 
                content_type="text/plain", 
                status=status.HTTP_400_BAD_REQUEST
            )

        # Fetch API details for the given deployment model
        api_details = APIDetails.objects.filter(deployment_model=deployment_model).values().first()

        if not api_details:
            return HttpResponse(
                f"No API details found for deployment model '{deployment_model}'.", 
                content_type="text/plain", 
                status=status.HTTP_404_NOT_FOUND
            )

        # Define the configuration file path
        config_file_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'project', 'taskweaver_config.json'))

        # Create the configuration dictionary
        config = {
            "llm.api_base": api_details['api_endpoint'],
            "llm.api_type": api_details['api_type'],
            "llm.model": api_details['deployment_model'],
            "llm.response_format": api_details['response_format'],
            "llm.azure.api_version": api_details['api_version'],
            "llm.past_messages": api_details['past_messages'],
            "llm.max_tokens": api_details['max_tokens'],
            "llm.temperature": api_details['temperature'],
            "llm.top_p": api_details['top_p'],
            "llm.frequency_penalty": api_details['frequency_penalty'],
            "llm.presence_penalty": api_details['presence_penalty'],
            "llm.stop": api_details['stop'],
            "llm.stream": api_details['stream'],
            "llm.azure_ad.aad_auth_mode": "default_azure_credential",
            "execution_service.kernel_mode": "local"
        }

        # Write the config data to the JSON file
        with open(config_file_path, 'w') as file:
            json.dump(config, file, indent=4)

        return HttpResponse(
            "taskweaver_config.json updated successfully.",
            content_type="text/plain",
            status=status.HTTP_200_OK
        )

    except Exception as e:
        # Handle unexpected errors
        return HttpResponse(
            f"An error occurred: {str(e)}",
            content_type="text/plain",
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
def create_api_details(request):
    """
    Creates a new API details entry in the APIDetails model.

    Args:
        request: The HTTP POST request containing the API details data.

    Returns:
        Response: A JSON response with the created API details or validation errors.
    """
    try:
        # Initialize the serializer with the request data
        serializer = APIDetailsSerializer(data=request.data)

        # Validate the data and save if valid
        if serializer.is_valid():
            serializer.save()  # Save the validated data to the APIDetails model
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        # If data is invalid, return errors
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    except Exception as e:
        # Handle unexpected errors
        return Response(
            {"error": f"An error occurred: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def update_api_details(request):
    """
    Updates an existing API details entry in the APIDetails model.

    Args:
        request: The HTTP POST request containing the updated API details data.

    Returns:
        Response: A JSON response with the updated API details or validation errors.
    """
    # Retrieve the ID from the request data
    id = request.data.get('id')

    # Check if ID is provided
    if not id:
        return Response({"error": "ID is required"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # Get the APIDetails instance by ID
        api_details = APIDetails.objects.get(id=id)
    except APIDetails.DoesNotExist:
        return Response({"error": "APIDetails with the given ID does not exist"}, status=status.HTTP_404_NOT_FOUND)

    # Deserialize and validate the request data for partial updates
    serializer = APIDetailsSerializer(api_details, data=request.data, partial=True)

    if serializer.is_valid():
        serializer.save()  # Save the updated data to the database
        return Response(serializer.data, status=status.HTTP_200_OK)

    # Return validation errors if any
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

def update_csv_file(plugin_name, updates):
    """Update the CSV file with plugin details based on the provided updates."""
    plugins_folder = os.path.abspath(os.path.join(os.getcwd(), '.'))
    csv_file_path = os.path.join(plugins_folder, 'plugins.csv')
    
    # Load CSV file if it exists, otherwise create a new DataFrame
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
    else:
        # If CSV doesn't exist, create an empty DataFrame with the expected columns
        df = pd.DataFrame(columns=['plugin_name', 'Endpoint', 'DEPLOYMENT_NAME'])
    
    # Ensure the 'plugin_name' column is available in the DataFrame
    if 'plugin_name' not in df.columns:
        raise ValueError("CSV file must contain a 'plugin_name' column.")
    
    # Extract values from the updates dictionary with default values
    endpoint = updates.get('endpoint_url', 'default_endpoint')
    deployment_name = updates.get('deployment_name', 'default_deployment_name')
    
    # Check if the plugin already exists in the CSV
    if plugin_name in df['plugin_name'].values:
        index = df[df['plugin_name'] == plugin_name].index
        if not index.empty:
            # Update the relevant fields
            df.at[index[0], 'Endpoint'] = endpoint
            df.at[index[0], 'DEPLOYMENT_NAME'] = deployment_name
    else:
        # If plugin does not exist, add it as a new row
        new_row = {
            'plugin_name': plugin_name,
            'Endpoint': endpoint,
            'DEPLOYMENT_NAME': deployment_name
        }
        df = df.append(new_row, ignore_index=True)
    
    # Save the DataFrame back to the CSV file
    df.to_csv(csv_file_path, index=False)

def convert_to_ast_node(value):
    """Converts a value to an appropriate AST node."""
    if isinstance(value, bool):
        return ast.Constant(value=value)
    elif isinstance(value, (int, float, str)):
        return ast.Constant(value=value)
    elif isinstance(value, list):
        # Convert list to an AST List node
        return ast.List(elts=[convert_to_ast_node(item) for item in value], ctx=ast.Store())
    elif isinstance(value, dict):
        # Convert dict to an AST Dict node
        keys = [convert_to_ast_node(key) for key in value.keys()]
        values = [convert_to_ast_node(val) for val in value.values()]
        return ast.Dict(keys=keys, values=values)
    elif isinstance(value, tuple):
        # Convert tuple to an AST Tuple node
        return ast.Tuple(elts=[convert_to_ast_node(item) for item in value], ctx=ast.Load())
    else:
        # For unsupported types, return as a constant (handle with care)
        return ast.Constant(value=str(value))  # Use string conversion as fallback

def update_python_file(plugin_name, updates):
    """Update the Python file variables based on the provided updates."""
    plugins_folder = os.path.abspath(os.path.join(os.getcwd(), '..', 'project', 'plugins'))
    python_file_path = os.path.join(plugins_folder, plugin_name) + ".py"
    
    if not os.path.exists(python_file_path):
        print(f"Python file not found: {python_file_path}")
        return

    try:
        with open(python_file_path, 'r') as file:
            code = file.read()
        
        # Parse the code into an Abstract Syntax Tree (AST)
        tree = ast.parse(code)
        
        # Replace the variables in the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):  # Look for assignment statements
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in updates:
                        # Replace the variable with the new value
                        new_value = updates[target.id]
                        # Convert the new value into a valid AST node
                        node.value = convert_to_ast_node(new_value)
        
        # Convert the updated AST back to source code
        updated_code = ast.unparse(tree)
        
        # Save the updated code back to the Python file
        with open(python_file_path, 'w') as file:
            file.write(updated_code)
        
        print(f"Python file updated successfully: {python_file_path}")

    except PermissionError:
        print(f"Permission denied while trying to update: {python_file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def update_YAML_files(plugin_name, updates):
    """Update the YAML file based on the provided JSON string."""
    plugins_folder = os.path.abspath(os.path.join(os.getcwd(), '..', 'project', 'plugins'))
    yaml_file_path = os.path.join(plugins_folder, plugin_name) + ".yaml"
    
    # Convert JSON string to a dictionary
    endpoint = updates.get('endpoint', 'default_endpoint')
    deployment_name = updates.get('deployment', 'default_deployment_name')

    # Modifying the payloads data
    payload_updates = {
        "endpoint_url": endpoint,
        "deployment_name": deployment_name
    }

    # Update YAML file
    if os.path.exists(yaml_file_path):
        try:
            with open(yaml_file_path, 'r') as file:
                data = yaml.safe_load(file)

            # Ensure the 'configurations' key exists
            if "configurations" not in data:
                logger.error(f"'configurations' key not found in {yaml_file_path}.")
                return f"Error: 'configurations' key not found in {yaml_file_path}."

            # Update data with matching keys from the updates dictionary
            for key, value in payload_updates.items():
                if key in data["configurations"]:
                    data["configurations"][key] = value
                else:
                    logger.warning(f"Key '{key}' not found in the YAML file. No update applied.")

            # Save the updated data back to the YAML file
            with open(yaml_file_path, 'w') as file:
                yaml.safe_dump(data, file)
            
            logger.info("YAML file updated successfully!")
            return "YAML file updated successfully!"

        except Exception as e:
            logger.error(f"Failed to update YAML file: {str(e)}")
            return f"Error updating YAML file: {str(e)}"
    else:
        logger.error(f"YAML file not found: {yaml_file_path}")
        return f"YAML file not found: {yaml_file_path}"

@api_view(['POST'])
def update_files(request):
    """Update the YAML file and CSV file based on the provided JSON string."""
    data = request.data
    plugin_name = data.get('name')
    
    if not plugin_name:
        return HttpResponse("Plugin name is required.", content_type="text/plain", status=status.HTTP_400_BAD_REQUEST)
    
    if not data:
        return HttpResponse("No update data provided.", content_type="text/plain", status=status.HTTP_400_BAD_REQUEST)

    try:
        # Update the CSV file with the provided updates
        update_csv_file(plugin_name, data)
    except Exception as e:
        return HttpResponse(f"Error updating CSV file: {str(e)}", content_type="text/plain", status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    try:
        # Update the YAML file with the provided updates
        update_YAML_files(plugin_name, data)
    except Exception as e:
        return HttpResponse(f"Error updating YAML file: {str(e)}", content_type="text/plain", status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    response_text = "CSV and YAML files updated successfully."
    return HttpResponse(response_text, content_type="text/plain", status=status.HTTP_200_OK)


@api_view(['GET'])
def configfile(request):
    """Returns the content of the taskweaver_config.json file."""
    config_file_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'project', 'taskweaver_config.json'))
    
    try:
        # Check if the file exists before attempting to open it
        if not os.path.exists(config_file_path):
            return HttpResponse(f"Config file not found at {config_file_path}", status=status.HTTP_404_NOT_FOUND)

        # Read current JSON file content and return as a response
        with open(config_file_path, 'r') as file:
            config = json.load(file)
        
        return JsonResponse(config, safe=False)
    
    except json.JSONDecodeError:
        # If the file contains invalid JSON, return an error message
        return HttpResponse("Error: Config file contains invalid JSON.", content_type="text/plain", status=status.HTTP_400_BAD_REQUEST)
    
    except Exception as e:
        # Generic error handler for any other issues (file reading, permissions, etc.)
        return HttpResponse(f"Error: {str(e)}", content_type="text/plain", status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_yaml_files_with_key_value(request):
    """
    Retrieves all YAML files in the plugins folder where the key 'custom_modification' is True.
    """
    plugins_folder_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'project', 'plugins'))
    target_key = 'custom_modification'
    matching_files = []

    # Iterate through all files in the directory
    for filename in os.listdir(plugins_folder_path):
        if filename.endswith((".yaml", ".yml")):  # Check both .yaml and .yml extensions
            file_path = os.path.join(plugins_folder_path, filename)
            try:
                # Load YAML file
                with open(file_path, 'r') as file:
                    data = yaml.safe_load(file)
                    
                    # Check if the key exists and its value is True
                    if data and data.get(target_key) is True:
                        matching_files.append(filename[:-5])  # Remove '.yaml' or '.yml' extension
            
            except yaml.YAMLError as e:
                # Log YAML parsing errors
                logger.error(f"Error reading YAML file {filename}: {e}")
            except Exception as e:
                # Log any unexpected errors
                logger.error(f"Unexpected error with file {filename}: {e}")

    # Return the matching files as a JSON response
    return JsonResponse(matching_files, safe=False)


@api_view(['POST'])
def create_session(request):
    try:
        # Check if 'user' is in the request data
        if 'user' not in request.data:
            return Response({'error': "'user' key is required in request data"}, status=status.HTTP_400_BAD_REQUEST)

        # Set plugin folders (ensure these are necessary and unique)
        code_executor.set_plugin_folder("plugins")
        set_plugin_folder("plugins")

        global user_session_id
        global app_session_dict
        global session

        # Retrieve user session id
        user_session_id = request.data['user']

        # Create or get session for the user
        app_session_dict[user_session_id] = app.get_session()
        session = app_session_dict[user_session_id]

        # Optionally retrieve additional session information, like the current working directory
        session_cwd_path = session.execution_cwd
        session.event_emitter  # Trigger event emitter if needed

        # Return success response with session ID and status
        return Response({
            'session_id': session.session_id,
            'status': 'Session created successfully',
            'execution_cwd': session_cwd_path  # You can optionally return the current working directory
        }, status=status.HTTP_201_CREATED)

    except KeyError as e:
        return Response({'error': f"Missing key: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
    
    except AttributeError as e:
        return Response({'error': f"Error in session attributes: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
    
    except Exception as e:
        return Response({'error': f"An unexpected error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    

@api_view(['POST'])
def create_draftplugin_session(request):
    try:
        # Validate that the 'user' key exists in the request data
        if 'user' not in request.data:
            return Response({'error': "'user' key is required in request data"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Set plugin folder paths (Ensure these are necessary)
        code_executor.set_plugin_folder("draftplugins")
        set_plugin_folder("draftplugins")

        global draft_session_id
        global draft_session_dict
        global draftsession

        # Retrieve user session ID
        draft_session_id = request.data['user']

        # Create or get session for the user
        draft_session_dict[draft_session_id] = draftapp.get_session()
        draftsession = draft_session_dict[draft_session_id]

        # Optionally retrieve additional session information, like current working directory
        draftsession_cwd_path = draftsession.execution_cwd
        draftsession.event_emitter  # Trigger event emitter if needed

        # Return success response with session ID and status
        return Response({
            'session_id': draftsession.session_id,
            'status': 'Session created successfully',
            'execution_cwd': draftsession_cwd_path  # You can optionally return the current working directory
        }, status=status.HTTP_201_CREATED)

    except KeyError as e:
        return Response({'error': f"Missing key: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
    
    except AttributeError as e:
        return Response({'error': f"Error in session attributes: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
    
    except Exception as e:
        return Response({'error': f"An unexpected error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def create_admin_session(request):
    try:
        # Validate that the 'user' key exists in the request data
        if 'user' not in request.data:
            return Response({'error': "'user' key is required in request data"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Set plugin folder paths (Ensure these are necessary)
        code_executor.set_plugin_folder("adminplugins")
        set_plugin_folder("adminplugins")

        global admin_session_id
        global admin_session_dict
        global adminsession

        # Retrieve user session ID
        admin_session_id = request.data['user']

        # Create or get session for the user
        admin_session_dict[admin_session_id] = adminapp.get_session()
        adminsession = admin_session_dict[admin_session_id]

        # Optionally retrieve additional session information, like current working directory
        adminsession_cwd_path = adminsession.execution_cwd
        adminsession.event_emitter  # Trigger event emitter if needed

        # Return success response with session ID and status
        return Response({
            'session_id': adminsession.session_id,
            'status': 'Admin session created successfully',
            'execution_cwd': adminsession_cwd_path  # You can optionally return the current working directory
        }, status=status.HTTP_201_CREATED)

    except KeyError as e:
        return Response({'error': f"Missing key: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
    
    except AttributeError as e:
        return Response({'error': f"Error in session attributes: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
    
    except Exception as e:
        return Response({'error': f"An unexpected error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

import logging
from datetime import datetime

class ConsoleEventHandler(SessionEventHandler):
    def handle(self, event: TaskWeaverEvent):
        try:
            # Log the event with timestamp and message
            self.logger.info(f"Event Time: {event.t}, Event Message: {event.msg}")
        except AttributeError as e:
            # Handle case where expected attributes are missing
            self.logger.error(f"Error processing event: {str(e)}")


@api_view(['POST'])
def SendMessage(request):
    global session
    encoded_blob = ''
    img_path = ''
    data_url = ''
    mime_type = ''
    file_name = ''
    user_msg_content = ''
    
    # Deserialize the data
    serializer = BlobDataSerializer(data=request.data)
    prompt = request.data.get("prompt")  # Safely accessing 'prompt'
    
    if serializer.is_valid():
        encoded_blob = serializer.validated_data['blob_data']
        file_name = serializer.validated_data['File_name']
        
        try:
            # Decode the base64 encoded blob and save it to a file
            decoded_data = base64.b64decode(encoded_blob)
            file_path = os.path.join('..', 'Temp_file', file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'wb') as f:
                f.write(decoded_data)
            
            file_path = os.path.realpath(file_path)
            files_to_send = [{"name": file_name, "path": file_path}]
        
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Handle the message and send it using the session
        if prompt:
            res = session.send_message(message=prompt, event_handler=ConsoleEventHandler(), files=files_to_send)
        
        # Extract artifact paths and prepare for attachment if available
        artifact_paths = [
            p for p in res.post_list for a in p.attachment_list
            if a.type == AttachmentType.artifact_paths for p in a.content
        ]
        
        session_cwd_path = session.execution_cwd
        
        for post in [p for p in res.post_list if p.send_to == "User"]:
            files = []
            if artifact_paths:
                for file_path in artifact_paths:
                    img_path = file_path
                    file_name = os.path.basename(file_path)
                    files.append((file_name, file_path))
            
            user_msg_content = post.message
            pattern = r"(!?)\[(.*?)\]\((.*?)\)"
            matches = re.findall(pattern, user_msg_content)
            
            # Process markdown image URLs and replace them in the message
            for match in matches:
                img_prefix, file_name, file_path = match
                if not img_path:
                    img_path = os.path.join(session_cwd_path, file_path)
                
                files.append((file_name, file_path))
                user_msg_content = user_msg_content.replace(f"{img_prefix}[{file_name}]({file_path})", file_name)
                
                # Encode the image as base64
                with open(img_path, 'rb') as file:
                    file_data = file.read()
                encoded_blob = base64.b64encode(file_data).decode('utf-8')
                mime_type, _ = mimetypes.guess_type(img_path)
                data_url = f'data:{mime_type};base64,{encoded_blob}'
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    
    # Return response with the processed message and file details
    return JsonResponse({'content': user_msg_content, 'encoded_blob': data_url, 'file_type': mime_type, 'file_name': file_name}, safe=False)


@api_view(['POST'])
def draftPluginSendMessage(request):
    global draftsession
    encoded_blob = ''
    img_path = ''
    data_url = ''
    mime_type = ''
    file_name = ''
    user_msg_content = ''
    
    # Deserialize the data
    serializer = BlobDataSerializer(data=request.data)
    prompt = request.data.get("prompt")  # Access 'prompt' safely
    
    if serializer.is_valid():
        encoded_blob = serializer.validated_data['blob_data']
        file_name = serializer.validated_data['File_name']
        
        try:
            # Decode the base64 encoded blob and save it to a file
            decoded_data = base64.b64decode(encoded_blob)
            file_path = os.path.join('..', 'Temp_file', file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'wb') as f:
                f.write(decoded_data)
            
            file_path = os.path.realpath(file_path)
            files_to_send = [{"name": file_name, "path": file_path}]
        
        except Exception as e:
            return Response({'error': f"File handling error: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Send message with or without the file
            res = draftsession.send_message(message=prompt, event_handler=ConsoleEventHandler(), files=files_to_send if prompt else None)
            
            # Extract artifact paths from the response
            artifact_paths = [
                p for p in res.post_list
                for a in p.attachment_list
                if a.type == AttachmentType.artifact_paths
                for p in a.content
            ]
            
            draftsession_cwd_path = draftsession.execution_cwd
            for post in [p for p in res.post_list if p.send_to == "User"]:
                files = []
                if artifact_paths:
                    for file_path in artifact_paths:
                        img_path = file_path
                        file_name = os.path.basename(file_path)
                        files.append((file_name, file_path))
                
                user_msg_content = post.message
                pattern = r"(!?)\[(.*?)\]\((.*?)\)"
                matches = re.findall(pattern, user_msg_content)
                
                # Process markdown image URLs and replace them in the message
                for match in matches:
                    img_prefix, file_name, file_path = match
                    if not img_path:
                        img_path = os.path.join(draftsession_cwd_path, file_path)
                    
                    files.append((file_name, file_path))
                    user_msg_content = user_msg_content.replace(f"{img_prefix}[{file_name}]({file_path})", file_name)
                    
                    # Encode the image as base64
                    with open(img_path, 'rb') as file:
                        file_data = file.read()
                    encoded_blob = base64.b64encode(file_data).decode('utf-8')
                    mime_type, _ = mimetypes.guess_type(img_path)
                    data_url = f'data:{mime_type};base64,{encoded_blob}'
            
        except Exception as e:
            return Response({'error': f"Message sending error: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
    
    return JsonResponse({'content': user_msg_content, 'encoded_blob': data_url, 'file_type': mime_type, 'file_name': file_name}, safe=False)


@api_view(['POST'])
def adminSendMessage(request):
    global adminsession
    global yaml_code_content
    global python_code_content
    global plugin_name
    global plugin_description
    
    encoded_blob = ''
    img_path = ''
    data_url = ''
    mime_type = ''
    file_name = ''
    user_msg_content = ''
    
    # Deserialize the data
    serializer = BlobDataSerializer(data=request.data)
    prompt = request.data.get("prompt")  # Access 'prompt' safely
    
    # Check if the serializer is valid
    if serializer.is_valid():
        encoded_blob = serializer.validated_data['blob_data']
        file_name = serializer.validated_data['File_name']
        
        try:
            # Decode the base64 encoded blob and save it to a file
            decoded_data = base64.b64decode(encoded_blob)
            file_path = os.path.join('..', 'Temp_file', file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'wb') as f:
                f.write(decoded_data)
            
            file_path = os.path.realpath(file_path)
            files_to_send = [{"name": file_name, "path": file_path}]
        
        except Exception as e:
            return Response({'error': f"File handling error: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Send message with or without the file
            res = adminsession.send_message(message=prompt, event_handler=None, files=files_to_send if prompt else None)
            
            # Extract artifact paths from the response
            artifact_paths = [
                p for p in res.post_list
                for a in p.attachment_list
                if a.type == AttachmentType.artifact_paths
                for p in a.content
            ]
            
            adminsession_cwd_path = adminsession.execution_cwd
            for post in [p for p in res.post_list if p.send_to == "User"]:
                files = []
                if artifact_paths:
                    for file_path in artifact_paths:
                        img_path = file_path
                        file_name = os.path.basename(file_path)
                        files.append((file_name, file_path))
                
                user_msg_content = post.message

            # Extract Python and YAML code from the message
            python_pattern = r'```python\n(.*?)```'
            yaml_pattern = r'```yaml\n(.*?)```'
            python_code = re.search(python_pattern, user_msg_content, re.DOTALL)
            yaml_code = re.search(yaml_pattern, user_msg_content, re.DOTALL)

            # Check and extract plugin details
            if not python_code_content:
                python_code_content = python_code.group(1).strip() if python_code else ""
                yaml_code_content = yaml_code.group(1).strip() if yaml_code else ""
                
                if yaml_code_content:
                    match = re.search(r'(plugin_name|name):\s*(\S+)', yaml_code_content)
                    plugin_description_match = re.search(r'(plugin_description|description):\s*(\S+)', yaml_code_content)
                    
                    if match:
                        plugin_name = match.group(2)
                    if plugin_description_match:
                        plugin_description = plugin_description_match.group(2)

            # Check if the plugin creation conditions are met
            if (plugin_name in user_msg_content or "new plugin" in user_msg_content) and \
               "success" in user_msg_content and ("saved" in user_msg_content or "created" in user_msg_content or "submit" in user_msg_content):
                # Write the plugin files to the draft plugin directory
                plugin_path = os.path.realpath("../project/draftplugins")
                with open(os.path.join(plugin_path, f"{plugin_name}.py"), "w") as py_file:
                    py_file.write(python_code_content)
                with open(os.path.join(plugin_path, f"{plugin_name}.yaml"), "w") as yaml_file:
                    yaml_file.write(yaml_code_content)
                
                # Generate intent catalog for the plugin
                generate_intent_catalog(plugin_name, plugin_description)
                
                # Reset global variables after successful plugin creation
                yaml_code_content = ""
                python_code_content = ""
                plugin_name = ""
                plugin_description = ""

        except Exception as e:
            return Response({'error': f"Message sending error: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
    
    return JsonResponse({'content': user_msg_content, 'encoded_blob': data_url, 'file_type': mime_type, 'file_name': file_name}, safe=False)


@api_view(['POST'])
def delete_plugin(request):
    plugin_name = request.data.get('plugin_name')  # Correct spelling of plugin_name
    if not plugin_name:
        return Response({"error": "Plugin name is required"}, status=status.HTTP_400_BAD_REQUEST)
    
    plugins_folder_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'project', 'plugins'))
    plugin_file_name = f"{plugin_name}.yaml"
    file_path = os.path.join(plugins_folder_path, plugin_file_name)

    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            return Response({"error": f"Plugin file '{plugin_file_name}' not found"}, status=status.HTTP_404_NOT_FOUND)
        
        # Load YAML file
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        
        if not data:
            return Response({"error": f"Failed to read data from {plugin_file_name}"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Update the target keys
        data["isdeleted"] = True
        data["enabled"] = False
        
        # Save the updated data back to the YAML file
        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file, indent=4)
        
        return JsonResponse({"message": f"Plugin '{plugin_name}' marked as deleted successfully."}, safe=False)
    
    except yaml.YAMLError as e:
        return Response({"error": f"Error processing YAML file: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({"error": f"Unexpected error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def approve_plugin(request):
    plugin_name = request.data.get('plugin_name')
    approve_plugin = request.data.get('approve')
    
    if not plugin_name:
        return Response({"error": "Plugin name is required"}, status=status.HTTP_400_BAD_REQUEST)
    
    # Define paths
    plugins_folder_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'project', 'draftplugins'))
    plugin_yaml_name = f"{plugin_name}.yaml"
    py_name = f"{plugin_name}.py"
    
    file_path = os.path.join(plugins_folder_path, plugin_yaml_name)
    py_file_path = os.path.join(plugins_folder_path, py_name)
    
    target_plugins_folder_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'project', 'plugins'))
    yaml_target_path = os.path.join(target_plugins_folder_path, plugin_yaml_name)
    py_target_path = os.path.join(target_plugins_folder_path, py_name)

    try:
        # Check if the draft files exist
        if not os.path.exists(file_path):
            return Response({"error": f"Plugin YAML file '{plugin_yaml_name}' not found"}, status=status.HTTP_404_NOT_FOUND)
        
        if not os.path.exists(py_file_path):
            return Response({"error": f"Plugin Python file '{py_name}' not found"}, status=status.HTTP_404_NOT_FOUND)
        
        # If the plugin is being approved, copy the files
        if approve_plugin == "Approve":
            shutil.copyfile(file_path, yaml_target_path)
            shutil.copyfile(py_file_path, py_target_path)
        
        # Load the YAML file for update
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        
        if not data:
            return Response({"error": "Failed to read plugin data"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Update the YAML file to reflect the approval
        if approve_plugin == "Approve":
            data["isdeleted"] = False  # Set to False since it's being approved
            data["enabled"] = True     # Enable the plugin
        
        # Save the updated YAML file back
        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file, indent=4)
        
        return JsonResponse({"message": f"Plugin '{plugin_name}' has been approved successfully."}, safe=False)
    
    except yaml.YAMLError as e:
        return Response({"error": f"Error processing YAML file: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({"error": f"Unexpected error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def update_packages(request):
    if request.method == "POST":
        try:
            # Parse the JSON data from the request body
            data = json.loads(request.body)
            print(data)
            
            # Validate input
            if not isinstance(data, list) or not all("name" in pkg for pkg in data):
                return JsonResponse({"error": "Invalid data format. Expected a list of {'name': ..., 'version': ... (optional)}."}, status=400)
            
            # Define file path for the requirements.txt file
            file_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'requirements.txt'))
            
            # Ensure the file exists before appending
            if not os.path.exists(file_path):
                open(file_path, "w").close()

            # Read existing data to avoid duplicates
            with open(file_path, mode="r") as file:
                existing_lines = {line.strip() for line in file.readlines()}

            # Prepare new lines to append
            new_lines = []
            for package in data:
                name = package["name"]
                version = package.get("version", None)
                
                # If no version is provided, assume a default version (>=1.0.0)
                if version:
                    line = f"{name}>={version}"
                else:
                    line = name  # No version specified
                
                if line not in existing_lines:  # Avoid duplicate entries
                    new_lines.append(line)

            # Append new lines to the file
            if new_lines:
                with open(file_path, mode="a") as file:
                    for line in new_lines:
                        file.write(f"{line}\n")

                return JsonResponse({"message": "requirements.txt updated successfully", "file_path": file_path}, status=200)
            else:
                return JsonResponse({"message": "No new packages to add."}, status=200)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON data."}, status=400)
        except OSError as e:
            return JsonResponse({"error": f"File error: {str(e)}"}, status=500)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Only POST requests are allowed."}, status=405)


def generate_intent_catalog(plugin_name, plugin_description):
    # Load configuration from the file
    config_file_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'project', 'taskweaver_config.json'))
    
    try:
        with open(config_file_path, 'r') as config_file:
            config = json.load(config_file)
    except Exception as e:
        return JsonResponse({"error": f"Failed to load configuration: {str(e)}"}, status=500)

    # Extract necessary values from the config
    api_base = config.get("llm.api_base")
    api_version = config.get("llm.azure.api_version")
    model_name = config.get("llm.model")
    api_type = config.get("llm.api_type")

    # Ensure parameters are valid numbers or set defaults
    try:
        temperature = float(config.get("llm.temperature", 0.7))
        top_p = float(config.get("llm.top_p", 1.0))
        frequency_penalty = float(config.get("llm.frequency_penalty", 0.0))
        presence_penalty = float(config.get("llm.presence_penalty", 0.0))
    except ValueError as e:
        return JsonResponse({"error": f"Invalid parameter value: {str(e)}"}, status=400)

    # Authenticate using DefaultAzureCredential
    try:
        credential = DefaultAzureCredential()

        # Prepare the prompt for LLM
        prompt = f"""
            Generate 20 possible intents for the given plugin. Only provide the intent names.

            Plugin Name: {plugin_name}
            Plugin Description: {plugin_description}
        """

        # Create the AzureChatOpenAI client
        client = AzureChatOpenAI(
            azure_endpoint=api_base,
            openai_api_version=api_version,
            deployment_name=model_name,
            openai_api_type=api_type,
            azure_ad_token_provider=lambda: credential.get_token("https://cognitiveservices.azure.com/.default").token,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        # Get the response from the model
        response = client(prompt)
        
        if not response.content:
            return JsonResponse({"error": "No response content returned from the model."}, status=500)
        
        intents = response.content.strip().split('\n')
        intent_string = "\n".join(intents)

        # Prepare file path for the intent catalog CSV
        intent_file_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'project', 'Essential_Documents', 'intent_routing', 'Intent_catalogue.csv'))

        # Ensure the directory and file exist
        os.makedirs(os.path.dirname(intent_file_path), exist_ok=True)
        
        # Check if the file exists; if not, create the header
        if not os.path.exists(intent_file_path):
            with open(intent_file_path, 'w', newline='', encoding='utf-8') as intent_file:
                intent_file.write("Plugin Name, Intents\n")  # Write the CSV header

        # Append the new intents to the CSV
        with open(intent_file_path, 'a', newline='', encoding='utf-8') as intent_file:
            intent_file.write(f"{plugin_name},\"{intent_string}\"\n")

        return JsonResponse({"response": intents}, status=200)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
