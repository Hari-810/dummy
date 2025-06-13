import nest_asyncio
import asyncio
from gremlin_python.driver import client

# Patch event loop (for Jupyter async compatibility)
nest_asyncio.apply()

# Connect to Gremlin Server
GREMLIN_ENDPOINT = 'ws://localhost:8182/gremlin'
gclient = client.Client(GREMLIN_ENDPOINT, 'g')

# Function to submit and print results
def run_query(query_str):
    print("Running Query:\n", query_str.strip())
    try:
        result_set = gclient.submit(query_str)
        results = result_set.all().result()
        for r in results:
            print(r)
    except Exception as e:
        print(" Error:", e)

# ##### GQL

query = "g.V().hasLabel('Engagement').valueMap(true)"
run_query(query)

# When you're done, close the client
# gclient.close()

# Check if WorkItems exist
# run_query("g.V().hasLabel('WorkItem').valueMap(true)")

# Check if Risks exist
# run_query("g.V().hasLabel('Risk').valueMap(true)")

# Check if Documents exist
# run_query("g.V().hasLabel('Document').valueMap(true)")

#run_query("g.E().label().dedup()")

# run_query("g.E().groupCount().by(label)")

# run_query("""
# g.E().project('from', 'to', 'label')
#   .by(outV().values('uuid'))
#   .by(inV().values('uuid'))
#   .by(label)
# """)

#Find all WorkItems for a given Engagement ID - 41BC9B5F-57AB-4673-FF64-08DB5AAE0E07
run_query("""
g.V().has('Engagement', 'uuid', '41BC9B5F-57AB-4673-FF64-08DB5AAE0E07')
  .out('hasWorkItem')
  .valueMap(true)
""")

#Get all Risks & Documents linked to an Engagement
run_query("""
g.V().has('Engagement', 'uuid', '41BC9B5F-57AB-4673-FF64-08DB5AAE0E07')
  .both('hasRisk', 'hasDocument')
  .valueMap(true)
""")

run_query("g.V().label().dedup()")

run_query("g.V().hasLabel('Risk').has('InkRiskNumber', 'RAITCOR0011').valueMap(true)")

#run_query("g.V().hasLabel('Engagement').outE().as('edge').inV().as('to').select('edge', 'to').by(label).by(valueMap(true))")



# #### Retrieve PIpeline

import pandas as pd
import chromadb
import os
import tiktoken
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI
import uuid
from chromadb.config import Settings
import re
import json

from gremlin_python.process.traversal import TextP

import nest_asyncio
import asyncio
from gremlin_python.driver import client

# Patch event loop (for Jupyter async compatibility)
nest_asyncio.apply()

# Connect to Gremlin Server
GREMLIN_ENDPOINT = 'ws://localhost:8182/gremlin'
gclient = client.Client(GREMLIN_ENDPOINT, 'g')

# Function to submit and print results
def run_query(query_str):
    print("Running Query:\n", query_str.strip())
    try:
        result_set = gclient.submit(query_str)
        results = result_set.all().result()
        for r in results:
            print(r)
    except Exception as e:
        print(" Error:", e)

data = """ 
All labels below Engagement, person, Document, software, Risk, WorkItem

There Lables and its properties/columns
1. Engagement (columns: ["Id", "ClientEngagementId", "ClientId", "Name"])
2. WorkItem (columns: ["Id", "EngagementId", "TypeId", "ReferenceNumber", "Name", "DisplayName", "Description"])
3. Risk (columns: ["Id", "EngagementId", "InkRiskNumber", "RiskNumber", "Description", "ClassificationTypeId", "InkSuggestedClassificationTypeId", "RiskTypeId", "RiskOriginTypeId"])
4. Document (columns: ["Id", "FileId", "EngagementId", "WorkItemId", "FileName", "DownloadUrl", "FileSize", "FileType"])
"""


prompt = """
You are a Gremlin Query Generator.

Your task is to generate a valid Gremlin query in JSON format based on the provided graph-based schema `{data}` and a user question `{query}`.

---

### Inputs:
1. **table_schema** (type: dict)  
   - A dictionary where:
     - Each **key** is a label (node type).
     - Each **value** is a list of column/property names for that label.
   - It defines the graph schema based on converted relational structure.

2. **user_query** (type: string)  
   - A natural language question from the user about the data.

---

### Instructions:

1. Parse the `user_query` to identify target labels and columns.
2. Use `{data}` to:
   - Determine which label(s) each field belongs to.
   - Detect connections using common fields (`EngagementId`, `WorkItemId`) and infer edge paths (like `hasRisk`, `hasWorkItem`, etc.).
3. Generate a valid **Gremlin Groovy** query:
   - Use `.hasLabel('<Label>')` to filter node type.
   - Use `.has('<column>', '<value>')` for value matching.
   - Use `.out('<edge>')` / `.in('<edge>')` to traverse related nodes.
   - Use `.project(...).by(...)` for multi-column selection.
   - For ordering, use `Order.asc` or `Order.desc` (Groovy-compatible).
   - For prefix match, use `TextP.startingWith("...")`
   - Avoid non-Gremlin-standard prefixes like `text.` – they are not valid in Groovy.
4. If filtering numerics (like FileSize > 1000000), use `P.gt(1000000)`.
5. For queries that involve filtering a parent node (e.g., WorkItem) based on the properties of a related node (e.g., Document), follow this Gremlin pattern:
    - Use `.hasLabel('<ParentLabel>').as('alias')`
    - Traverse with `.out('<edgeName>')` to the related node (child)
    - Apply property filters (e.g., `.has('FileName', containing('ERA'))`) on the related node
    - Use `.select('alias')` to return to the parent node
    - Then use `.project(...).by(...)` to select desired properties of the parent

6. Avoid direct value comparisons across two traversals using `eq()` in Groovy. Instead, use label aliasing (`.as()`) and `.select()` to maintain context and apply filters.

7. When using `TextP` predicates like `containing`, `startingWith`, etc., always assume:
    - `import static org.apache.tinkerpop.gremlin.process.traversal.TextP.*` is already available
    - Use lowercase methods like `containing('value')` directly (not `TextP.containing` or `text.containing`).

8. Never reference property values like `'alias.property'` inside `.where()` or `eq()`. Always use `.select('alias')` then continue traversal.

9. For sorting, use `.order().by('<property>', Order.asc)` or `Order.desc` (not `asc` or `decr`).

10. Maintain this idiomatic Gremlin structure:
    - `.hasLabel(...)`
    - `.as(...)`
    - `.out(...)`
    - `.has(...)`
    - `.select(...)`
    - `.project(...).by(...)`

11. Do not use unsupported methods like `eq()`, `text.contains()`, or `select('property')` without aliasing — these are not valid in Gremlin Groovy.

12. carefully analyse in and out below is the relationship
Engagement → WorkItem (hasWorkItem)
Engagement → Risk (hasRisk)
Engagement → Document (hasDocument)
WorkItem → Document (hasDocument)
---

### Output Format:
Return your answer in the following **JSON format**:

```json
{{
                
   "gremlin_query": "<GREMLIN QUERY>",
   "description": "<Short explanation of the query>"
                
}}
"""

token_provider = get_bearer_token_provider(
            DefaultAzureCredential(logging_enable=True),
            "https://cognitiveservices.azure.com/.default"
        )

llm_model = AzureChatOpenAI(
        openai_api_version="2024-02-15-preview",
        azure_deployment="gpt-4o-mini-v2024-07-18-ptu",
        azure_endpoint="https://aimlameuse2npdopenai.openai.azure.com/",
        azure_ad_token_provider=token_provider
    )

def extract_json_from_response(response_text: str):
        """
        Safely extracts JSON from an LLM response even if it's wrapped in Markdown or contains additional text.
        Detects truncation and logs appropriately.
        """
        try:
            # First: try to extract Markdown-style JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Fallback: try to extract direct JSON array
                json_match = re.search(r'(\[\s*{.*?}\s*\])', response_text, re.DOTALL)
                json_str = json_match.group(1) if json_match else response_text.strip()

            # Optional: basic truncation check
            if not json_str.strip().endswith(("]", "}")):
                print("Possible output truncation detected: JSON does not end with expected character.")

            # Optional: attempt to auto-close the array (only if it *looks* like it's truncated)
            if json_str.count("{") > json_str.count("}"):
                print("Attempting to fix incomplete JSON object.")
                json_str += "}" * (json_str.count("{") - json_str.count("}"))

            if json_str.count("[") > json_str.count("]"):
                print("Attempting to fix incomplete JSON array.")
                json_str += "]" * (json_str.count("[") - json_str.count("]"))

            return json.loads(json_str)

        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Failed content: {json_str[:500]}...")
            return []   

# ##### Test-1

query = "show RiskNumber where the column InkRiskNumber is equal to RAITCOR0011"
formatted_prompt = prompt.format(data=data,query=query)
response1 = llm_model.invoke(formatted_prompt)
res=extract_json_from_response(response1.content)
print(res['gremlin_query'])

run_query(res['gremlin_query'])

# ##### Test-2

#query = "write a query to retrieve all information in Risk label for the column InkRiskNumber and its value RAITCOR0011"
query = "Extract all FileName where the DisplayName is GITC-00416 GITC 165EFA09D"
formatted_prompt = prompt.format(data=data,query=query)
response1 = llm_model.invoke(formatted_prompt)
res=extract_json_from_response(response1.content)
print(res['gremlin_query'])

run_query(res['gremlin_query'])

# ##### Test-3

query = "give all the filename start with ENG"
formatted_prompt = prompt.format(data=data,query=query)
response1 = llm_model.invoke(formatted_prompt)
res=extract_json_from_response(response1.content)
print(res['gremlin_query'])

run_query(res['gremlin_query'])

# ##### Test-4

query = "give all the Risk id and InkRiskNumber where the descriptions starts Income tax expense"
formatted_prompt = prompt.format(data=data,query=query)
response1 = llm_model.invoke(formatted_prompt)
res=extract_json_from_response(response1.content)
print(res['gremlin_query'])

run_query(res['gremlin_query'])

# ##### Test-5

query = "give all InkRiskNumber where the descriptions related to capital expenditures"
formatted_prompt = prompt.format(data=data,query=query)
response1 = llm_model.invoke(formatted_prompt)
res=extract_json_from_response(response1.content)
print(res['gremlin_query'])

run_query(res['gremlin_query'])

# ##### Test-6

query = "give all download url and its DisplayName GITC-00416 GITC 165EFA09D"
formatted_prompt = prompt.format(data=data,query=query)
response1 = llm_model.invoke(formatted_prompt)
res=extract_json_from_response(response1.content)
print(res['gremlin_query'])

run_query(res['gremlin_query'])

# ##### Test-7

query = "show all data of work item names which contain 'GITC' keyword"
formatted_prompt = prompt.format(data=data,query=query)
response1 = llm_model.invoke(formatted_prompt)
res=extract_json_from_response(response1.content)
print(res['gremlin_query'])

run_query(res['gremlin_query'])

# ##### Test-8

#show unstructured docs along with work item details where document name contains "x" word

#query = "show  all workitem details where the FileName contains 'ERA' word"
query = "show all document data where the FileName contains 'ERA' word"
formatted_prompt = prompt.format(data=data,query=query)
response1 = llm_model.invoke(formatted_prompt)
res=extract_json_from_response(response1.content)
print(res['gremlin_query'])

run_query(res['gremlin_query'])

# ##### Test-9

#show unstructured docs along with work item details where document name contains "x" word

query = "show all workitem details where the FileName contains 'ERA' word, where id matches"
formatted_prompt = prompt.format(data=data,query=query)
response1 = llm_model.invoke(formatted_prompt)
res=extract_json_from_response(response1.content)
print(res['gremlin_query'])

run_query(res['gremlin_query'])

run_query("""g.V().hasLabel('WorkItem').as('wi')
  .out('hasDocument')
  .hasLabel('Document')
  .has('FileName', containing('ERA'))
  .select('wi')
  .project('Id', 'EngagementId', 'TypeId', 'ReferenceNumber', 'Name', 'DisplayName', 'Description')
    .by('Id')
    .by('EngagementId')
    .by('TypeId')
    .by('ReferenceNumber')
    .by('Name')
    .by('DisplayName')
    .by('Description')""")
# run_query("g.E().count()")

# run_query("g.V().drop()")

# run_query("g.V().count()")
# run_query("g.E().count()")

