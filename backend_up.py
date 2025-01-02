import os
import sys
import re
import ast
import csv
import base64
import json
import yaml
import atexit
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from azure.identity import DefaultAzureCredential

from .models import LLMModel, APIDetails
from .serializers import APIDetailsSerializer, BlobDataSerializer

import msal
import requests

from taskweaver.app.app import TaskWeaverApp  # Import TaskWeaverApp
from taskweaver.memory.attachment import AttachmentType
from taskweaver.memory.type_vars import RoleName
from taskweaver.module.event_emitter import PostEventType, RoundEventType, SessionEventHandlerBase
from taskweaver.session.session import Session

# Load environment variables
load_dotenv()

# Set up repository and project paths
repo_path = os.path.join(os.path.dirname(__file__), "../../")
sys.path.append(repo_path)

project_path = os.path.join(repo_path, "project")

# Initialize TaskWeaverApp
app = TaskWeaverApp(app_dir=project_path, use_local_uri=True)
atexit.register(app.stop)

# Initialize session dictionary
app_session_dict: Dict[str, Session] = {}
user_session_id = ""
app_session_dict[user_session_id] = app.get_session()

# Placeholder session variable
session = ""

@api_view(['GET'])
def is_authorized(request):
    """
    Description:
        Fetches username from the frontend. Based on MSAL, an access token is fetched. 
        This token is used to retrieve member information of a specific group using a Graph API URL. 
        If the username matches a UserPrincipalName in the group members, the function returns True; 
        otherwise, it returns False.
    
    Parameters:
        request (HttpRequest): Used to retrieve the username from headers.
    
    Returns:
        JsonResponse: A JSON object in the format {"Access": True/False}.
    """
    # Retrieve username from headers
    username = request.headers.get('username')
    if not username:
        return JsonResponse({"error": "Username header is missing."}, status=400)
    
    # Configuration
    client_id = os.getenv("ClientId")
    client_secret = os.getenv("ClientSecret")
    authority = os.getenv("Instance") + os.getenv("TenantId")
    group_id = os.getenv("GroupId_1")
    scope = ['https://graph.microsoft.com/.default']
    
    if not all([client_id, client_secret, authority, group_id]):
        return JsonResponse({"error": "Missing environment configuration."}, status=500)
    
    config = {
        'client_id': client_id,
        'client_secret': client_secret,
        'authority': authority,
        'scope': scope
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
        return JsonResponse({
            "error": "Failed to acquire access token.",
            "details": token_result.get('error_description')
        }, status=500)
    
    # Fetch group members
    url = f'https://graph.microsoft.com/v1.0/groups/{group_id}/members'
    headers = {'Authorization': f"Bearer {token_result['access_token']}"}
    azure_group_members = []
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        azure_group = response.json()
        azure_group_members.extend(azure_group.get('value', []))
        
        # Handle pagination if applicable
        while '@odata.nextLink' in azure_group:
            next_url = azure_group['@odata.nextLink']
            response = requests.get(next_url, headers=headers)
            response.raise_for_status()
            azure_group = response.json()
            azure_group_members.extend(azure_group.get('value', []))
    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": "Failed to fetch group members.", "details": str(e)}, status=500)
    
    # Check username
    is_username_present = any(user.get("userPrincipalName") == username for user in azure_group_members)
    return JsonResponse({"Access": is_username_present}, safe=False)

@api_view(['GET'])
def list_plugins(request):
    """
    Retrieves a list of plugins from the plugins folder, including their names,
    descriptions, examples, parameters, return values, and enabled statuses.

    This function reads each `.yaml` file in the plugins folder and extracts
    details such as `plugin_name`, `description`, `examples`, `parameters`,
    `return values`, and `enabled_status`. If a field is missing, it provides a
    default message. It also updates a CSV file with these details.

    Returns:
        JsonResponse: A list of dictionaries containing plugin details.
    """
    plugins_folder_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'project', 'plugins'))
    csv_file_path = 'plugins.csv'

    # Ensure the plugins folder exists
    if not plugins_folder_path.exists():
        return JsonResponse({"error": "Plugins folder not found."}, status=404)

    # Get all .yaml files from the plugins folder
    yaml_files = [f for f in plugins_folder_path.iterdir() if f.suffix == '.yaml' and f.is_file()]

    # Load existing plugin details from CSV
    existing_plugins = {}
    if csv_file_path.exists():
        with csv_file_path.open('r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                existing_plugins[row['plugin_name']] = row

    plugins = []
    for yaml_file in yaml_files:
        try:
            with yaml_file.open('r') as file:
                data = yaml.safe_load(file)

            plugin_name = yaml_file.stem
            description = data.get('description', 'No description available')
            enabled_status = data.get('enabled', 'No enabled status available')
            example = data.get('examples', 'No example available')
            custom_modification = data.get('custom_modification', False)

            configurations = data.get('configurations', {})
            Endpoint = configurations.get('endpoint_url', 'No Endpoint available')
            DEPLOYMENT_NAME = configurations.get('deployment_name', 'No deployment name available')

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
            plugin_info = {
                "plugin_name": plugin_name,
                "description": description,
                "example": example,
                "parameter_description": parameter_description,
                "return_description": return_description,
                "enabled_status": enabled_status,
                "custom_modification": custom_modification,
                "Endpoint": Endpoint,
                "DEPLOYMENT_NAME": DEPLOYMENT_NAME,
            }
            plugins.append(plugin_info)

            # Update or add to existing plugins
            existing_plugins[plugin_name] = plugin_info

        except Exception as e:
            return JsonResponse({"error": f"Error processing {yaml_file.name}: {str(e)}"}, status=500)

    # Write updated plugin details back to the CSV
    try:
        with csv_file_path.open('w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=[
                "plugin_name", "description", "example", "parameter_description",
                "return_description", "enabled_status", "custom_modification",
                "Endpoint", "DEPLOYMENT_NAME"
            ])
            writer.writeheader()
            writer.writerows(existing_plugins.values())
    except Exception as e:
        return JsonResponse({"error": f"Failed to write to CSV: {str(e)}"}, status=500)

    return JsonResponse(plugins, safe=False)

@api_view(['POST'])
def plugin_enable(request):
    """
    Enables or disables a plugin by updating its YAML configuration file.

    Args:
        request (HttpRequest): The request containing 'plugin_name' and 'state' in JSON format.

    Returns:
        JsonResponse: A response indicating the success or failure of the operation.
    """
    data = request.data
    plugin_name = data.get('plugin_name')
    plugin_state = data.get('state')

    # Validate input
    if not plugin_name or plugin_state is None:
        return JsonResponse(
            {"error": "Both 'plugin_name' and 'state' must be provided."},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        # Define the path to the plugins folder and the YAML file
        plugins_folder =  os.path.abspath(os.path.join(os.getcwd(), '..', 'project', 'plugins'))
        yaml_file = plugins_folder / f"{plugin_name}.yaml"

        if not yaml_file.exists():
            return JsonResponse(
                {"error": f"Plugin file '{plugin_name}.yaml' not found."},
                status=status.HTTP_404_NOT_FOUND
            )

        # Load the YAML file
        with yaml_file.open('r') as file:
            plugin_data = yaml.safe_load(file)

        # Update the 'enabled' field based on the state
        plugin_data['enabled'] = bool(plugin_state)

        # Save the updated YAML file
        with yaml_file.open('w') as file:
            yaml.dump(plugin_data, file, default_flow_style=False)

        return JsonResponse(
            {"message": "YAML file updated successfully.", "plugin_name": plugin_name, "state": plugin_state},
            status=status.HTTP_200_OK
        )

    except Exception as e:
        return JsonResponse(
            {"error": f"An error occurred: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
