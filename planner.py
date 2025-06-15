import nest_asyncio
import asyncio
import pandas as pd
from gremlin_python.driver import client
import uuid

# Patch event loop (for Jupyter async compatibility)
nest_asyncio.apply()


# ------------------------------------------------------------------------------
# 1. GREMLIN CONSTANTS & INITIALIZE
# ------------------------------------------------------------------------------

GREMLIN_ENDPOINT = 'ws://localhost:8182/gremlin'
gclient = client.Client(GREMLIN_ENDPOINT, 'g')

# ------------------------------------------------------------------------------
# 2. HELPER FUNCTIONS FOR ADDING DATA
# ------------------------------------------------------------------------------

def escape_string(value):
    """Escape backslashes and quotes in strings to safely insert into gremlin."""
    if isinstance(value, str):
        return value.replace("\\", "\\\\").replace("'", "\\'")
    return str(value)

def add_vertex_query(label, props):
    """Generate a gremlin query to add a vertex with given properties."""
    query = f"g.addV('{label}')"
    query += f".property('uuid', '{escape_string(props['Id'])}')"
    query += f".property('label', '{label}')"

    for k, v in props.items():
        if k == 'Id' or pd.isnull(v):
            continue
        safe_key = escape_string(k)
        safe_val = escape_string(v)
        query += f".property('{safe_key}', '{safe_val}')"

    return query

def add_edge_query(from_label, from_id, to_label, to_id, edge_label):
    """Generate a gremlin query to connect two vertices."""
    return f"""g.V().has('{from_label}', 'uuid', '{from_id}').as('a')
               .V().has('{to_label}', 'uuid', '{to_id}').as('b')
               .addE('{edge_label}').from('a').to('b')"""    

def submit_gremlin(gclient, query):
    """Submit a gremlin query safely and handle exception if fails."""
    try:
        gclient.submit(query).all().result()
    except Exception as e:
        print(f"Error executing: {query}\nError: {e}")

def run_query(gclient, query_str):
    """Submit and execute a gremlin read query and return results."""
    try:
        result_set = gclient.submit(query_str)
        results = result_set.all().result()
        return results
    except Exception as e:
        print(f"Error executing: {query_str}\nError: {e}")
        return []

# ------------------------------------------------------------------------------
# 3. DATA Loading Section
# ------------------------------------------------------------------------------

def load_data(eng_file, work_file, risk_file, doc_file):
    """Load CSV or Excel files into pandas DataFrames."""
    df_engagement = pd.read_excel(eng_file)
    df_workitem = pd.read_excel(work_file)
    df_risks = pd.read_excel(risk_file)
    df_docs = pd.read_excel(doc_file)

    return df_engagement, df_workitem, df_risks, df_docs

def populate_graph(gclient, df_engagement, df_workitem, df_risks, df_docs):
    """Add all data to graph and create relationships."""
    for _, row in df_engagement.iterrows():
        props = row.to_dict()
        query = add_vertex_query("Engagement", props)
        submit_gremlin(gclient, query)

    for _, row in df_workitem.iterrows():
        props = row.to_dict()
        query = add_vertex_query("WorkItem", props)
        submit_gremlin(gclient, query)
        if pd.notna(row["EngagementId"]):
            edge_query = add_edge_query("Engagement", row["EngagementId"], "WorkItem", row["Id"], "hasWorkItem")
            submit_gremlin(gclient, edge_query)

    for _, row in df_risks.iterrows():
        props = row.to_dict()
        query = add_vertex_query("Risk", props)
        submit_gremlin(gclient, query)
        if pd.notna(row["EngagementId"]):
            edge_query = add_edge_query("Engagement", row["EngagementId"], "Risk", row["Id"], "hasRisk")
            submit_gremlin(gclient, edge_query)

    for _, row in df_docs.iterrows():
        props = row.to_dict()
        query = add_vertex_query("Document", props)
        submit_gremlin(gclient, query)
        if pd.notna(row["EngagementId"]):
            edge_query = add_edge_query("Engagement", row["EngagementId"], "Document", row["Id"], "hasDocument")
            submit_gremlin(gclient, edge_query)
        if pd.notna(row["WorkItemId"]):
            edge_query = add_edge_query("WorkItem", row["WorkItemId"], "Document", row["Id"], "hasDocument")
            submit_gremlin(gclient, edge_query)

# ------------------------------------------------------------------------------
# 4. Main
# ------------------------------------------------------------------------------

def main():
    """ Main pipeline """
    engagement_file = "AI_Engagement 1.xlsx"
    workitem_file = "AI_WorkItem 1.xlsx"
    risks_file = "AI_Risks 1.xlsx"
    documents_file = "AI_UnstructuredDocs 1.xlsx"

    df_engagement, df_workitem, df_risks, df_docs = load_data(engagement_file, workitem_file, risks_file, documents_file)

    populate_graph(gclient, df_engagement, df_workitem, df_risks, df_docs)

    # ------------------------------------------------------------------------------
    # Run some sample queries
    # ------------------------------------------------------------------------------

    # 1. Retrieve all engagements
    results = run_query(gclient, "g.V().hasLabel('Engagement').valueMap(true)")
    for r in results:
        print(r)

    # 2. Retrieve all risks
    results = run_query(gclient, "g.V().hasLabel('Risk').valueMap(true)")
    for r in results:
        print(r)

    # 3. Retrieve all documents
    results = run_query(gclient, "g.V().hasLabel('Document').valueMap(true)")
    for r in results:
        print(r)

    gclient.close()


if __name__ == "__main__":
    main()
