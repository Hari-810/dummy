import pytest
import io
from httpx import AsyncClient, ASGITransport
from rag_api import app

# Configure reusable ASGI transport
transport = ASGITransport(app=app)


@pytest.mark.asyncio
async def test_health_check():
    """Basic test to check if health endpoint is alive."""
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()


@pytest.mark.asyncio
async def test_sync_textual_file():
    """Valid case: Sync a textual file."""
    data = {'data_category': 'Textual', 'sync_to_kb': 'true'}
    files = [('files', ('sample.txt', io.BytesIO(b"This is a test document."), 'text/plain'))]
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.post("/rag/sync", data=data, files=files)
        assert response.status_code == 200
        json_resp = response.json()
        assert "success" in json_resp
        assert "message" in json_resp


@pytest.mark.asyncio
async def test_sync_tabular_file():
    """Valid case: Sync a CSV tabular file."""
    data = {'data_category': 'Tabular', 'sync_to_kb': 'true'}
    files = [('files', ('sample.csv', io.BytesIO(b"col1,col2\nval1,val2"), 'text/csv'))]
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.post("/rag/sync", data=data, files=files)
        assert response.status_code == 200
        json_resp = response.json()
        assert "success" in json_resp
        assert "filenames" in json_resp


@pytest.mark.asyncio
async def test_sync_file_missing_file():
    """Edge case: Missing uploaded file."""
    data = {'data_category': 'Textual', 'sync_to_kb': 'true'}
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.post("/rag/sync", data=data)
        assert response.status_code in [400, 422]


@pytest.mark.asyncio
async def test_sync_file_invalid_filetype():
    """Edge case: Unsupported file type uploaded."""
    data = {'data_category': 'Textual', 'sync_to_kb': 'true'}
    files = [('files', ('sample.exe', io.BytesIO(b"fake binary"), 'application/octet-stream'))]
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.post("/rag/sync", data=data, files=files)
        assert response.status_code in [400, 422]


@pytest.mark.asyncio
async def test_retrieve_documents_textual():
    """Valid case: Retrieve documents from textual data."""
    payload = {"query": "example document content", "top_k": 2, "data_category": "Textual"}
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.post("/rag/retrieve", json=payload)
        assert response.status_code == 200
        json_resp = response.json()
        assert "chunks" in json_resp
        assert isinstance(json_resp["chunks"], list)


@pytest.mark.asyncio
async def test_retrieve_documents_tabular_with_schema():
    """Valid case: Retrieve tabular documents with schema metadata."""
    payload = {
        "query": "synthetic sales data",
        "top_k": 3,
        "data_category": "Tabular",
        "schema_details": {
            "date": {"field_description": "The date of the sale"},
            "amount": {"field_description": "Total amount of the transaction"}
        }
    }
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.post("/rag/retrieve", json=payload)
        assert response.status_code == 200
        json_resp = response.json()
        assert "chunks" in json_resp
        assert isinstance(json_resp["chunks"], list)
        assert "success" in json_resp


@pytest.mark.asyncio
async def test_retrieve_documents_missing_query():
    """Edge case: Missing 'query' field in retrieval payload."""
    payload = {"top_k": 3, "data_category": "Textual"}
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.post("/rag/retrieve", json=payload)
        assert response.status_code in [400, 422]


@pytest.mark.asyncio
async def test_retrieve_documents_empty_query():
    """Edge case: Empty query string provided."""
    payload = {"query": "", "top_k": 2, "data_category": "Textual"}
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.post("/rag/retrieve", json=payload)
        assert response.status_code == 200
        json_resp = response.json()
        assert "chunks" in json_resp  # Could be empty list
        assert isinstance(json_resp["chunks"], list)


@pytest.mark.asyncio
async def test_retrieve_documents_invalid_category():
    """Edge case: Invalid data_category provided."""
    payload = {"query": "test", "top_k": 2, "data_category": "Audio"}
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.post("/rag/retrieve", json=payload)
        assert response.status_code in [400, 422]

"""

| **Test Function**                             | **Category**         | **Description**                                                                    |
| --------------------------------------------- | -------------------- | ---------------------------------------------------------------------------------- |
| `test_health_check`                           | Health               | Checks if the `/health` endpoint returns a 200 status with `"status"` in response. |
| `test_sync_textual_file`                      | Sync (Textual)       | Validates syncing of a valid `.txt` file under `"Textual"` category.               |
| `test_sync_tabular_file`                      | Sync (Tabular)       | Validates syncing of a valid `.csv` file under `"Tabular"` category.               |
| `test_sync_file_missing_file`                 | Sync (Edge Case)     | Tests syncing request with no files attached. Expects 400/422 status.              |
| `test_sync_file_invalid_filetype`             | Sync (Edge Case)     | Tests syncing unsupported file types (e.g., `.exe`). Expects 400/422 status.       |
| `test_retrieve_documents_textual`             | Retrieve (Textual)   | Validates retrieving chunks from textual data given a query.                       |
| `test_retrieve_documents_tabular_with_schema` | Retrieve (Tabular)   | Validates retrieving tabular data using schema details.                            |
| `test_retrieve_documents_missing_query`       | Retrieve (Edge Case) | Tests retrieval with missing `"query"` key. Expects 400/422 status.                |
| `test_retrieve_documents_empty_query`         | Retrieve (Edge Case) | Tests retrieval with empty query string. Returns 200 with empty or valid chunks.   |
| `test_retrieve_documents_invalid_category`    | Retrieve (Edge Case) | Tests retrieval using an unsupported `data_category` like `"Audio"`.               |


"""

