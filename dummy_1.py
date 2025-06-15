from fastapi import APIRouter, File, UploadFile, HTTPException
import pandas as pd
from uuid import uuid4
from typing import List
from app.services.gremlin.client import submit_gremlin

router = APIRouter()


def create_node_gremlin_query(label: str, row: dict) -> str:
    """Create gremlin addV query for a node."""
    gremlin = f"g.addV('{label}')"
    for k, v in row.items():
        if isinstance(v, str):
            gremlin += f".property('{k}', '{v}')"
        else:
            gremlin += f".property('{k}', {v})"
    return gremlin


def connect_nodes(from_label: str, from_id_field: str, to_label: str, to_id_field: str, edge_name: str) -> str:
    """Create gremlin addE query to connect two nodes by IDs."""
    return f"""g.V().has('{from_label}', 'Id', '{from_id_field}').as('a')
                 .V().has('{to_label}', 'Id', '{to_id_field}').addE('{edge_name}').from('a') """


@router.post("/load/engagements")
async def load_engagements(file: UploadFile = File(...)):
    """Load engagement data from CSV or Excel into graph."""
    try:
        df = pd.read_csv(file.file) if file.filename.endswith('.csv') else pd.read_excel(file.file)
    except Exception as e:
        raise HTTPException(400, f"Invalid file format or read error: {e}")
    
    successfully_inserted = 0
    
    for _, row in df.iterrows():
        row_data = row.to_dict()
        row_data["Id"] = str(uuid4())  # generate unique IDs
        gremlin = create_node_gremlin_query("Engagement", row_data)
        submit_gremlin(gremlin)
        successfully_inserted += 1
    
    return {"total": successfully_inserted, "message": "Engagements successfully loaded."}


@router.post("/load/workitems")
async def load_workitems(file: UploadFile = File(...)):
    """Load WorkItems and connect them to their Engagements if EngagementId is present."""
    try:
        df = pd.read_csv(file.file) if file.filename.endswith('.csv') else pd.read_excel(file.file)
    except Exception as e:
        raise HTTPException(400, f"Invalid file format or read error: {e}")
    
    successfully_inserted = 0
    
    for _, row in df.iterrows():
        row_data = row.to_dict()
        row_data["Id"] = str(uuid4())  # generate unique IDs
        gremlin = create_node_gremlin_query("WorkItem", row_data)
        submit_gremlin(gremlin)

        if "EngagementId" in row_data and row_data["EngagementId"]:
            edge = connect_nodes("Engagement", row_data["EngagementId"], "WorkItem", row_data["Id"], "hasWorkItem")
            submit_gremlin(edge)

        successfully_inserted += 1
    
    return {"total": successfully_inserted, "message": "WorkItems successfully loaded."}


@router.post("/load/risks")
async def load_risks(file: UploadFile = File(...)):
    """Load Risks and connect them to their Engagements if EngagementId is present."""
    try:
        df = pd.read_csv(file.file) if file.filename.endswith('.csv') else pd.read_excel(file.file)
    except Exception as e:
        raise HTTPException(400, f"Invalid file format or read error: {e}")
    
    successfully_inserted = 0
    
    for _, row in df.iterrows():
        row_data = row.to_dict()
        row_data["Id"] = str(uuid4())  # generate unique IDs
        gremlin = create_node_gremlin_query("Risk", row_data)
        submit_gremlin(gremlin)

        if "EngagementId" in row_data and row_data["EngagementId"]:
            edge = connect_nodes("Engagement", row_data["EngagementId"], "Risk", row_data["Id"], "hasRisk")
            submit_gremlin(edge)

        successfully_inserted += 1
    
    return {"total": successfully_inserted, "message": "Risks successfully loaded."}


@router.post("/load/documents")
async def load_documents(file: UploadFile = File(...)):
    """Load documents and connect them to their Engagement or WorkItem if IDs are present."""
    try:
        df = pd.read_csv(file.file) if file.filename.endswith('.csv') else pd.read_excel(file.file)
    except Exception as e:
        raise HTTPException(400, f"Invalid file format or read error: {e}")
    
    successfully_inserted = 0
    
    for _, row in df.iterrows():
        row_data = row.to_dict()
        row_data["Id"] = str(uuid4())  # generate unique IDs
        gremlin = create_node_gremlin_query("Document", row_data)
        submit_gremlin(gremlin)

        if "EngagementId" in row_data and row_data["EngagementId"]:
            edge = connect_nodes("Engagement", row_data["EngagementId"], "Document", row_data["Id"], "hasDocument")
            submit_gremlin(edge)

        if "WorkItemId" in row_data and row_data["WorkItemId"]:
            edge = connect_nodes("WorkItem", row_data["WorkItemId"], "Document", row_data["Id"], "hasDocument")
            submit_gremlin(edge)

        successfully_inserted += 1
    
    return {"total": successfully_inserted, "message": "Documents successfully loaded."}

