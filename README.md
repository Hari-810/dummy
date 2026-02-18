# Simple GCP ADK Agent

## Files
- `gcp_simple_agent/agent.py`: root ADK agent
- `gcp_simple_agent/.env.example`: required GCP env vars

## Setup (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item gcp_simple_agent\.env.example gcp_simple_agent\.env
# Edit gcp_simple_agent\.env with your project/location
```

## Authenticate to GCP
```powershell
gcloud auth application-default login
```

## Run
From the repo root (`d:\hari\github\dummy`):
```powershell
adk run gcp_simple_agent
```

Or start the local web UI:
```powershell
adk web
```
Then select `gcp_simple_agent` in the UI.
