from pyspark.sql import SparkSession
import os
import time
import random
import json
from pyspark.dbutils import DBUtils

# Initialize Spark session
spark = SparkSession.builder.appName("ADLS_PDF_Processing").getOrCreate()
dbutils = DBUtils(spark)

# ADLS Configuration
storage_account_name = os.environ["STORAGE_ACCOUNT_NAME"]
container_name = os.environ["STORAGE_CONTAINER_NAME"]
mount_point = f"/mnt/{storage_account_name}/{container_name}"

# SPN Credentials
spn_tenant_id = os.environ["tenantId"]
spn_client_id = os.environ["clientId"]
spn_secret_scope = os.environ["clientSecretscope"]
spn_secret_key = os.environ["clientSecretName"]
spn_client_secret = dbutils.secrets.get(scope=spn_secret_scope, key=spn_secret_key)

# OAuth 2.0 Configurations
configs = {
    "fs.azure.account.auth.type": "OAuth",
    "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
    "fs.azure.account.oauth2.client.id": spn_client_id,
    "fs.azure.account.oauth2.client.secret": spn_client_secret,
    "fs.azure.account.oauth2.client.endpoint": f"https://login.microsoftonline.com/{spn_tenant_id}/oauth2/token"
}

# Mount ADLS (if not already mounted)
if not any(m.mountPoint == mount_point for m in dbutils.fs.mounts()):
    dbutils.fs.mount(
        source=f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/",
        mount_point=mount_point,
        extra_configs=configs
    )
    print(f"ADLS successfully mounted at {mount_point}")
else:
    print(f"ADLS is already mounted at {mount_point}")

# Get configurations from widgets
environment = os.environ["SYS_ENVIRONMENT"]
dbutils.widgets.get("configurations")
dbutils.widgets.get("configurations")
parsed_data = json.loads(dbutils.widgets.get("configurations"))
base_file_path = parsed_data["baseFilePath"]
upload_folder = f"/mnt/{storage_account_name}/{environment}/{base_file_path}/Uploads/"

def check_file_path_valid(file_path):
    """Check if a file path is valid."""
    return bool(file_path and isinstance(file_path, str))

def move_file(src_path, dest_path):
    """Move file from Uploads to Downloads folder."""
    try:
        dbutils.fs.mv(src_path, dest_path)
        return True
    except Exception as e:
        print(f"Error moving file: {e}")
        return False

def process_all_files(upload_folder):
    """Process all files from the Uploads folder and move them to the Downloads folder."""
    try:
        files = dbutils.fs.ls(upload_folder)
        time.sleep(random.randint(0, 300))  # Random delay before processing
        
        for file in files:
            file_path = os.path.join(upload_folder, file.name)
            if check_file_path_valid(file_path):
                move_file(file_path, file_path.replace("Uploads", "Downloads"))
    except Exception as e:
        print(f"Error processing files: {e}")

process_all_files(upload_folder)

# Response Data
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
        "msg": {"outputFileName": "brick_1_output.json"}
    }
}

# Print JSON response
print(json.dumps(data, indent=4))
