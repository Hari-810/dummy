import nest_asyncio
import asyncio
from gremlin_python.driver import client
import pandas as pd
import json
import re
import os
import tiktoken
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import uuid
# from chromadb.config import Settings
from gremlin_python.process.traversal import TextP

# ------------------------------------------------------------------------------
# 1️⃣ GREMLIN + CSV Loading Section (from Code 2)

# Patch event loop (for Jupyter async compatibility or reuse if already applied).
nest_asyncio.apply()

# ------------------------------------------------------------------------------
# GREMLIN SERVER ENDPOINT
GREMLIN_ENDPOINT = 'ws://localhost:8182/gremlin'
gclient = client.Client(GREMLIN_ENDPOINT, 'g')

def submit_gremlin(gremlin):
    """Submit gremlin to the database safely."""
    try:
        gclient.submit(gremlin).all().result()
    except Exception as e:
        print(f" Error: {e}")

def escape_string(value):
    """Escape backslashes and quotes in the string."""
    if isinstance(value, str):
        return value.replace("\\", "\\\\").replace("'", "\\'")
    return str(value)

def add_vertex(label, props):
    """Create a gremlin query to add a Vertex with properties."""
    query = f"g.addV('{label}')"
    query += f".property('uuid', '{escape_string(props['Id'])}')"
    query += f".property('label', '{label}')"

    for k, v in props.items():
        if k == 'Id' or pd.isna(v):
            continue
        query += f".property('{escape_string(k)}', '{escape_string(v)}')"

    return query

def add_edge(from_label, from_id, to_label, to_id, edge_label):
    """Create a gremlin query to add an edge."""
    return f"""g.V().has('{from_label}', 'uuid', '{from_id}').as('a')
                 .V().has('{to_label}', 'uuid', '{to_id}').as('b')
                 .addE('{edge_label}').from('a').to('b') """

def load_data(eng_file, work_file, risk_file, doc_file):
    """Load CSV files into graph database."""
    df_engagement = pd.read_excel(eng_file)
    df_workitem = pd.read_excel(work_file)
    df_risks = pd.read_excel(risk_file)
    df_docs = pd.read_excel(doc_file)

    for _, row in df_engagement.iterrows():
        props = row.to_dict()
        query = add_vertex("Engagement", props)
        submit_gremlin(query)

    for _, row in df_workitem.iterrows():
        props = row.to_dict()
        query = add_vertex("WorkItem", props)
        submit_gremlin(query)
        if pd.notna(row["EngagementId"]):
            edge_query = add_edge("Engagement", row["EngagementId"], "WorkItem", row["Id"], "hasWorkItem")
            submit_gremlin(edge_query)

    for _, row in df_risks.iterrows():
        props = row.to_dict()
        query = add_vertex("Risk", props)
        submit_gremlin(query)
        if pd.notna(row["EngagementId"]):
            edge_query = add_edge("Engagement", row["EngagementId"], "Risk", row["Id"], "hasRisk")
            submit_gremlin(edge_query)

    for _, row in df_docs.iterrows():
        props = row.to_dict()
        query = add_vertex("Document", props)
        submit_gremlin(query)
        if pd.notna(row["EngagementId"]):
            edge_query = add_edge("Engagement", row["EngagementId"], "Document", row["Id"], "hasDocument")
            submit_gremlin(edge_query)
        if pd.notna(row["WorkItemId"]):
            edge_query = add_edge("WorkItem", row["WorkItemId"], "Document", row["Id"], "hasDocument")
            submit_gremlin(edge_query)

# ------------------------------------------------------------------------------
# 2️⃣ OpenAI + Query Generation Section (from Code 1)

def extract_json_from_response(response_text: str):
    """Extract the first JSON object from a large text."""
    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text

        if json_str.count("{") > json_str.count("}"):
            json_str += "}" * (json_str.count("{") - json_str.count("}"))

        if json_str.count("[") > json_str.count("]"):
            json_str += "]" * (json_str.count("[") - json_str.count("]"))

        return json.loads(json_str)

    except Exception as e:
        print(f"JSON parsing failed: {e}")
        return []

# ------------------------------------------------------------------------------
# Prepare prompt and LLM

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

def generate_gremlin_query(user_query):
    """Send prompt to LLM and extract gremlin from its output."""
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(logging_enable=False),
        "https://cognitiveservices.azure.com/.default"
    )
    llm = AzureChatOpenAI(
        openai_api_version="2024-02-15-preview",
        azure_deployment="gpt-4o-mini-v2024-07-18-ptu",
        azure_endpoint="https://aimlameuse2npdopenai.openai.azure.com/",
        azure_ad_token_provider=token_provider
    )
    formatted_prompt = prompt.format(data=data, query=user_query)
    response = llm.invoke(formatted_prompt)
    res = extract_json_from_response(response.content)
    return res.get("gremlin_query")

def run_query(gremlin_query):
    """Execute a Gremlin query, handle both list/dict results, and remove duplicates."""
    print("Running Query:\n", gremlin_query)
    try:
        result_set = gclient.submit(gremlin_query)
        results = result_set.all().result()

        # Normalize results to a list
        if isinstance(results, dict):
            results = [results]

        elif not isinstance(results, list):
            results = [results]  # wrap single value in a list

        # Deduplication
        deduped_results = []
        seen = set()

        for r in results:
            # Convert dicts to a tuple of items so they're hashable
            key = tuple(r.items()) if isinstance(r, dict) else r
            if key not in seen:
                seen.add(key)
                deduped_results.append(r)

        # Print results
        print("results:\n")
        for item in deduped_results:
            print(item)

        return deduped_results

    except Exception as e:
        print("Error:", e)
        return []
# ------------------------------------------------------------------------------
# 3️⃣ Testing Section (sample questions)

def test():
    questions = [
        # "show RiskNumber where the column InkRiskNumber is equal to RAITCOR0011",
        "give all FileName where the DisplayName is GITC-00416 GITC 165EFA09D",
        # "give all Risk id and InkRiskNumber where the descriptions starts Income tax expense",
        # "give all download url and its DisplayName GITC-00416 GITC 165EFA09D",
        # "show all document data where the FileName contains 'ERA' word",
    ]

    for q in questions:
        print("*"*50)
        gremlin = generate_gremlin_query(q)
        print("Generated gremlin:\n", gremlin)
        run_query(gremlin)

# ------------------------------------------------------------------------------
# Main
if __name__ == "__main__":
    # Uncomment to execute:
    # load_data("AI_Engagement 1.xlsx", "AI_WorkItem 1.xlsx", "AI_Risks 1.xlsx", "AI_UnstructuredDocs 1.xlsx")
    test()
    gclient.close()
    print(" Done.")
