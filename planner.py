import pandas as pd
from gremlin_python.driver import client

GREMLIN_ENDPOINT = 'ws://localhost:8182/gremlin'

# Initialize Gremlin client
g_client = client.Client(GREMLIN_ENDPOINT, 'g')

# Read Excel files
df_engagement = pd.read_excel("AI_Engagement 1.xlsx")
df_workitem = pd.read_excel("AI_WorkItem 1.xlsx")
df_risks = pd.read_excel("AI_Risks 1.xlsx")
df_docs = pd.read_excel("AI_UnstructuredDocs 1.xlsx")

def submit_gremlin(query):
    try:
        g_client.submit(query).all().result()
    except Exception as e:
        print(f" Error: {e}")

def escape_string(value):
    if isinstance(value, str):
        return value.replace("\\", "\\\\").replace("'", "\\'")
    return str(value)

def add_vertex(label, props):
    # Start building query
    query = f"g.addV('{label}')"
    query += f".property('uuid', '{escape_string(props['Id'])}')"
    query += f".property('label', '{label}')"

    for k, v in props.items():
        if pd.notna(v) and k != 'Id':
            safe_key = escape_string(k)
            safe_val = escape_string(v)
            query += f".property('{safe_key}', '{safe_val}')"
    return query

def add_edge(from_label, from_id, to_label, to_id, edge_label):
    return f"""
    g.V().has('{from_label}', 'uuid', '{from_id}').as('a')
     .V().has('{to_label}', 'uuid', '{to_id}').as('b')
     .addE('{edge_label}').from('a').to('b')
    """

# Process Engagements
for _, row in df_engagement.iterrows():
    props = row.to_dict()
    query = add_vertex("Engagement", props)
    submit_gremlin(query)

# Process WorkItems and link to Engagement
for _, row in df_workitem.iterrows():
    props = row.to_dict()
    query = add_vertex("WorkItem", props)
    submit_gremlin(query)
    if pd.notna(row["EngagementId"]):
        edge_query = add_edge("Engagement", row["EngagementId"], "WorkItem", row["Id"], "hasWorkItem")
        submit_gremlin(edge_query)

# Process Risks and link to Engagement
for _, row in df_risks.iterrows():
    print("**********************")
    props = row.to_dict()
    query = add_vertex("Risk", props)
    submit_gremlin(query)
    if pd.notna(row["EngagementId"]):
        edge_query = add_edge("Engagement", row["EngagementId"], "Risk", row["Id"], "hasRisk")
        submit_gremlin(edge_query)

# Process Documents and link to Engagement and WorkItem (if available)
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

g_client.close()
print(" Data successfully loaded into the GraphDB.")
