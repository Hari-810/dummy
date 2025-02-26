from pyspark.sql import SparkSession
import time
import random 
from pyspark.dbutils import DBUtils
import json
import os
import time
import random
from pyspark.sql import SparkSession
import requests
from requests.exceptions import HTTPError


# Initialize Spark session
spark = SparkSession.builder.appName("ADLS_PDF_Processing").getOrCreate()
dbutils = DBUtils(spark)


# ADLS Configuration
storage_account_name = os.environ["STORAGE_ACCOUNT_NAME"]
container_name =   os.environ["STORAGE_CONTAINER_NAME"]
mount_point = f"/mnt/{storage_account_name}/{container_name}"

# SPN credentials
spn_tenant_id = os.environ["tenantId"]
spn_client_id = os.environ["clientId"]
spn_secret_scope = os.environ["clientSecretscope"] 
spn_secret_key =  os.environ["clientSecretName"]

spn_client_secret = dbutils.secrets.get(scope=spn_secret_scope, key=spn_secret_key)


# OAuth 2.0 Configurations 
configs = {
    "fs.azure.account.auth.type": "OAuth",
    "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
    "fs.azure.account.oauth2.client.id": spn_client_id,
    "fs.azure.account.oauth2.client.secret": spn_client_secret,
    "fs.azure.account.oauth2.client.endpoint": f"https://login.microsoftonline.com/{spn_tenant_id}/oauth2/token" 
}


# Mount ADLS (only if not already mounted)
if not any(m.mountPoint == mount_point for m in dbutils.fs.mounts()):
    dbutils.fs.mount(
        source=f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/",
        mount_point=mount_point,
        extra_configs=configs
    )
    print(f"ADLS successfully mounted at {mount_point}")
else:
    print(f"ADLS is already mounted at {mount_point}")



# Databricks job name from widget
json_string_00 = dbutils.widgets.get("requestInfo")
json_string = dbutils.widgets.get("configurations")
json_string_01 = dbutils.widgets.get("aimlRequest")


environment = os.environ["SYS_ENVIRONMENT"]
# Convert back to dictionary
parsed_data = json.loads(json_string)

# Extract values

baseFilePath = parsed_data["baseFilePath"]


file_path = f"/mnt/npdaimladlsuse2/{environment}/{baseFilePath}/Uploads/"
# print(file_path)




def check_file_path_valid(file_path):
    """Check if a file path is valid."""
    if not file_path or not isinstance(file_path, str):
        print("Invalid file path")
        return False

    return True

def move_file(src_path, dest_path):
    """Move file from Uploads to Downloads folder."""
    try:
        
        dbutils.fs.mv(src_path, dest_path)
        # print(f"File moved to {dest_path}")
        return True
    except Exception as e:
        print(f"Error moving file: {e}")
        return False

def process_all_files(upload_folder):
    """Process all files from the Uploads folder and move them to the Downloads folder."""
    try:
        files = dbutils.fs.ls(upload_folder)

        delay = random.randint(0, 300)
        # print(f"Sleeping for {delay} seconds before moving the file...")
        time.sleep(delay)

        for file in files:
            file_path = os.path.join(upload_folder, file.name)
            if check_file_path_valid(file_path):
                new_path = file_path.replace("Uploads", "Downloads")
                
                move_file(file_path, new_path)
    except Exception as e:
        print(f"Error processing files: {e}")

upload_folder = str(file_path)
process_all_files(upload_folder)
 




data = {
    "responseInfo": {
                "isSuccess": True,
                "ErrorMessage": "success",
                "tokenModels": [
                    {
                        "totalTokens": 27213,
                        "promptTokens": 22278,
                        "completionTokens": 4935,
                        "name": "gpt-4o",
                        "deploymentName": "gpt-4o-v2024-05-13",
                        "region": "eastus2"
                    }
                ]
            },
            "aimlResponse": {
                "version": "1.1.0",
                "msg": {
                    "outputFileName": "brick_1_output.json"
                }
            }
}

# Convert to JSON string and print
print(json.dumps(data, indent=4))
