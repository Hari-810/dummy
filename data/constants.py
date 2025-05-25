import os
from enum import Enum



# Define absolute paths for subdirectories inside 'data/'
DATA_FOLDER = os.path.abspath("Data")  
TABULAR_DATA_FOLDER = os.path.join(DATA_FOLDER, "tabular_synthetic_data")
TEXTUAL_DATA_FOLDER = os.path.join(DATA_FOLDER, "textual_synthetic_data")

# Config files
PROMPT_DIR = os.path.abspath("prompt_templates") 
TABULAR_PROMPT_YAML_PATH = os.path.join(PROMPT_DIR,"tabular_data.yaml")
TEXTUAL_PROMPT_YAML_PATH = os.path.join(PROMPT_DIR,"textual_data.yaml")

#Azure Openai variables:
class AZURE_MODELS_CONFIG:
    AZURE_OPENAI_ENDPOINT = "AZURE_OPENAI_ENDPOINT"
    API_VERSION = "API_VERSION"
    DEPLOYMENT_NAME = "DEPLOYMENT_NAME"
    TEMPERATURE=0.5
    BEARER_TOKEN_ENDPOINT="https://cognitiveservices.azure.com/.default"
    API_TYPE="azure"

# Model Token Limits for GPT-4o-mini
MODEL_INPUT_LIMIT = 128000  # 128K tokens (input limit for GPT-4o-mini)
MODEL_OUTPUT_LIMIT = 16384  # 16,384 tokens (output limit for GPT-4o-mini)

# Retry configuration for exponential backoff
RETRY_CONFIG = {
    "max_retries": 5,
    "initial_delay": 1, # in seconds
    "backoff_factor": 2
}

MAX_RETRIES = 5


# Ensure necessary directories exist before using them
os.makedirs(TABULAR_DATA_FOLDER, exist_ok=True)
os.makedirs(TEXTUAL_DATA_FOLDER, exist_ok=True)

# Data category:
class DATA_CATEGORY(Enum):
    TABULAR="tabular"
    TEXTUAL="textual"
    TIME_SERIES="Time-Series"
    IMAGE="Image"
    AUDIO="Audio"
    GRAPH="Graph"

class DATA_CLARITY(Enum):
    HIGH_FIDELITY="High Fidelity"
    NOISY="Noisy"
    RANDOM="Fully Randomized"


# Data categories and its required input configuration:
#---------------------------------Generic variables---------------------------------------
class GENERIC_VARIABLES:
    SAMPLE_COUNT="num_samples"
    USER_QUERY="user_query"
    REALISM_LEVEL="realism_level"
    DEFAULT_REALISM_LEVEL="High"
    DOMAIN="domain"
    DEFAULT_DOMAIN="Finance"
    OUTPUT_FORMAT="output_format"
    OF_CSV="CSV"
    OF_TEXT="TXT"
    OF_PARQUET="Parquet"
    OF_JSON="JSON"
    DATE_TIME_PATTERN="%Y%m%d_%H%M%S"

#---------------------------------Tabular data category-----------------------------------
class TDC_VARIABLES:
    YAML_FILE_NAME="synthetic_data_generation_prompt"
    SCHEMA_DETAILS="schema_details"
    FEATURE_TYPES="feature_types"
    DEFAULT_FEATURE_TYPES="Mixed"
    FEATURE_DISTRIBUTION="feature_distributions"
    DEFAULT_FEATURE_DISTRIBUTION="Normal"
    RELATIONSHIP_CONSTRAINTS="relationships_constraints"
    DEFAULT_OUTPUT_FORMAT="CSV"
    COLUMN_NAMES="column_names"
    FIELD_DESCRIPTION="field_description"
    MAX_LENGTH="max_length"
    USER_REQUEST="user_request"

#---------------------------------Natural Text data category-----------------------------
class NT_VARIABLES:
    YAML_FILE_NAME="synthetic_text_generation_prompt"
    TEXT_STYLE="text_style"
    DEFAULT_TEXT_STYLE="Narrative"
    TEXT_STRUCTURE="text_structure"
    DEFAULT_TEXT_STRUCTURE="Paragraph-Based"
    LOGICAL_CONSTRAINTS="logical_constraints"
    DEFAULT_OUTPUT_FORMAT="TXT"
    DEFAULT_SAMPLE_COUNT=1
    USER_REQUEST="user_request"
    DATA_CATEGORY="data_category"

