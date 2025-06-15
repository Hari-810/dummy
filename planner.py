To enhance **maintainability**, **modularity**, and **scalability** of your FastAPI project (with Gremlin, LLM, loaders, etc.), it's a good time to reorganize your folder structure following **best practices**.

---

### ✅ Suggested Folder Structure

```
your_project/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── api/                       # All route logic
│   │   ├── __init__.py
│   │   ├── endpoints/
│   │   │   ├── __init__.py
│   │   │   ├── query.py           # /query/
│   │   │   ├── loaders.py         # /load_*
│   │   │   ├── health.py          # /health/db
│   │
│   ├── services/                  # Business logic and orchestration
│   │   ├── __init__.py
│   │   ├── query_service.py       # Gremlin + LLM handler
│   │   ├── file_handler.py        # save_file, handle_file_upload
│   │
│   ├── db/                        # Gremlin database interaction
│   │   ├── __init__.py
│   │   ├── gremlin_manager.py
│   │   ├── gremlin_llm_executor.py
│   │
│   ├── loaders/                   # Excel/CSV loaders for ingestion
│   │   ├── __init__.py
│   │   ├── loaders.py             # load_engagements, etc.
│   │
│   ├── prompts/
│   │   ├── gremlin_prompt.yaml
│   │   ├── schema.txt
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── json_parser.py         # extract_json_from_response
│   │
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── openai_manager.py
│
├── temp_files/                    # Temporary upload storage
│
├── requirements.txt
└── README.md
```

---

### ✅ How to Refactor `main.py`

Refactor `main.py` to a clean router aggregator:

```python
# app/main.py
from fastapi import FastAPI
from db.gremlin_manager import GremlinManager
from contextlib import asynccontextmanager

from app.api.endpoints import query, loaders, health

@asynccontextmanager
async def lifespan(app: FastAPI):
    GremlinManager.connect()
    yield
    GremlinManager.close()

app = FastAPI(lifespan=lifespan)

# Include all routers
app.include_router(query.router)
app.include_router(loaders.router)
app.include_router(health.router)
```

---

### ✅ Split routes into endpoint files

#### `app/api/endpoints/query.py`

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.query_service import handle_query_from_input

router = APIRouter()

class QueryInput(BaseModel):
    query: str

@router.post("/query/", tags=["Query"])
def query_handler(request: QueryInput):
    try:
        response = handle_query_from_input(request.query)
        return {"status": "success", "data": response}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid input: {ve}")
    except ConnectionError as ce:
        raise HTTPException(status_code=503, detail=f"LLM service unavailable: {ce}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
```

---

#### `app/api/endpoints/loaders.py`

```python
from fastapi import APIRouter, File, UploadFile
from services.file_handler import handle_file_upload
from loaders.loaders import load_engagements, load_workitems, load_risks, load_docs

router = APIRouter()

@router.post("/load_engagements")
def import_engagements(file: UploadFile = File(...)):
    return handle_file_upload(file, load_engagements, "Engagements")

@router.post("/load_workitems")
def import_workitems(file: UploadFile = File(...)):
    return handle_file_upload(file, load_workitems, "WorkItems")

@router.post("/load_risks")
def import_risks(file: UploadFile = File(...)):
    return handle_file_upload(file, load_risks, "Risks")

@router.post("/load_docs")
def import_docs(file: UploadFile = File(...)):
    return handle_file_upload(file, load_docs, "Documents")
```

---

#### `app/api/endpoints/health.py`

```python
from fastapi import APIRouter
from db.gremlin_manager import GremlinManager

router = APIRouter()

@router.get("/health/db", tags=["Health Check"])
def check_db_health():
    try:
        test_query = "g.V().limit(1)"
        result_set = GremlinManager.submit(test_query)
        if result_set is None:
            return {"status": "unhealthy", "error": "Connection failed"}
        result = result_set.all().result()
        return {"status": "healthy", "result_count": len(result)}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

---

### ✅ Benefits of This Structure

| Benefit             | Description                                                    |
| ------------------- | -------------------------------------------------------------- |
| **Modularity**      | Clean separation of concerns (routes, logic, data access)      |
| **Testability**     | Services and DB functions are easily mockable                  |
| **Scalability**     | Adding new routes or LLM integrations is seamless              |
| **Reusability**     | Core logic (file handling, DB, LLM) is reused across endpoints |
| **Maintainability** | Easier debugging and onboarding for new developers             |

---

Let me know if you'd like a **starter GitHub repo** template or **unit test structure** too!




Great question! Helper functions (like file handling, string escaping, result formatting, etc.) should live in a **`utils/` (utilities)** or **`services/`** directory depending on their purpose.

---

### ✅ Where to Place Helper Functions

| Function Type                            | Suggested Location          | Reason |
|------------------------------------------|-----------------------------|--------|
| **General-purpose** (e.g., `escape_string`, JSON flattening, etc.) | `app/utils/`                | Shared across modules |
| **Business logic** (e.g., `handle_file_upload`) | `app/services/`             | Tied to app domain |
| **File I/O** (e.g., `save_file`, temp cleanup) | `app/services/file_handler.py` | Part of workflow services |

---

### ✅ Suggested Organization

#### `app/utils/string_utils.py`

```python
def escape_string(value):
    """Escape backslashes and single quotes in strings for Gremlin."""
    if isinstance(value, str):
        return value.replace("\\", "\\\\").replace("'", "\\'")
    return str(value)
```

---

#### `app/services/file_handler.py`

```python
import os
import shutil
import uuid
from fastapi import UploadFile
from fastapi import HTTPException

def save_file(file: UploadFile) -> str:
    """Save uploaded file to a temporary directory and return path."""
    directory = "temp_files"
    os.makedirs(directory, exist_ok=True)
    file_id = uuid.uuid4()
    ext = os.path.splitext(file.filename)[1]
    file_path = os.path.join(directory, f"{file_id}{ext}")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return file_path

def handle_file_upload(file: UploadFile, loader_fn, entity_name: str):
    """Generic handler for file upload and loader function."""
    try:
        path = save_file(file)
        loader_fn(path)
        os.remove(path)
        return {"status": "success", "message": f"{entity_name} loaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading {entity_name}: {e}")
```

---

#### `app/utils/json_utils.py`

If you're parsing LLM outputs like:
```python
def extract_json_block(text: str) -> dict:
    """Extract the first JSON block from a string (LLM response)."""
    import json, re
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found")
    return json.loads(match.group())
```

---

### ✅ How to Use in Modules

In `gremlin_manager.py`:
```python
from utils.string_utils import escape_string
```

In `loaders.py`, `query.py`, etc.:
```python
from services.file_handler import save_file, handle_file_upload
```

---

### 🔄 Optional Enhancements

You could later group helper files by domain, like:
```
utils/
├── string_utils.py
├── gremlin_utils.py      # gremlin-specific formatters, deduplication, aliasing
├── json_utils.py
```

---

Let me know if you want a `__init__.py` for each, or to convert `utils` into a package with testing!
