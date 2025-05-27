import json
import time
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication
from azure.devops.v7_1.work_item_tracking import Wiql
from azure.devops.v7_1.work_item_tracking import AttachmentReference

# ---------------------------
# Configuration
# ---------------------------
PERSONAL_ACCESS_TOKEN = "your_personal_access_token"
ORGANIZATION_URL = "https://dev.azure.com/symphonyvsts"
PROJECT_NAME = "Audit AIML"

# ---------------------------
# Authentication and Clients
# ---------------------------
def authenticate():
    credentials = BasicAuthentication("", PERSONAL_ACCESS_TOKEN)
    connection = Connection(base_url=ORGANIZATION_URL, creds=credentials)
    return connection

# ---------------------------
# Utility Functions
# ---------------------------
def parse_html(html_content):
    soup = BeautifulSoup(html_content or "", "html.parser")
    return {
        "text": soup.get_text(separator="\n"),
        "links": [a['href'] for a in soup.find_all('a', href=True)],
        "images": [img['src'] for img in soup.find_all('img', src=True)]
    }

def workitem_details(data):
    return {
        "id": data.get("id", ""),
        "title": data.get("title", ""),
        "description": data.get("description", ""),
        "work_item_type": data.get("work_item_type", ""),
        "acceptance_criteria": data.get("acceptance_criteria", ""),
        "business_outcome_hypothesis": data.get("business_outcome_hypothesis", "")
    }

# ---------------------------
# Data Extraction
# ---------------------------
def fetch_work_item_ids(wit_client):
    wiql_query = f"""
    SELECT [System.Id] FROM WorkItems
    WHERE [System.TeamProject] = '{PROJECT_NAME}'
    ORDER BY [System.Id] ASC
    """
    wiql = Wiql(query=wiql_query)
    result = wit_client.query_by_wiql(wiql=wiql)
    return [item.id for item in result.work_items]

def fetch_work_items(wit_client, work_item_ids, fields_to_fetch, batch_size=200):
    all_items = []
    for i in range(0, len(work_item_ids), batch_size):
        batch_ids = work_item_ids[i:i + batch_size]
        items = wit_client.get_work_items(ids=batch_ids, fields=fields_to_fetch)
        all_items.append(items)
    return all_items

def extract_work_items(all_items):
    records = []
    for items in all_items:
        for item in items:
            fields = item.fields
            try:
                records.append({
                    "id": int(fields.get("System.Id", "")),
                    "title": fields.get("System.Title", ""),
                    "work_item_type": fields.get("System.WorkItemType", ""),
                    "description": parse_html(fields.get("System.Description", "")),
                    "acceptance_criteria": parse_html(fields.get("Microsoft.VSTS.Common.AcceptanceCriteria", "")),
                    "business_outcome_hypothesis": fields.get("Custom.BusinessOutcomeHypothesis", ""),
                    "parent": fields.get("System.Parent", "")
                })
            except:
                pass
    return records

# ---------------------------
# Hierarchy Building
# ---------------------------
def convert_to_hierarchy(df):
    hierarchy = []
    epics = df[df['work_item_type'] == 'Epic']

    for _, epic in epics.iterrows():
        epic_block = {
            "epic_id": epic["id"],
            "epic_content": workitem_details(epic),
            "features": []
        }

        features = df[(df['work_item_type'] == 'Feature') & (df['parent'] == epic["id"])]
        for _, feature in features.iterrows():
            feature_block = {
                "feature_id": feature["id"],
                "feature_content": workitem_details(feature),
                "user_stories": []
            }

            user_stories = df[(df['work_item_type'] == 'User Story') & (df['parent'] == feature["id"])]
            for _, us in user_stories.iterrows():
                tasks = df[(df['work_item_type'] == 'Task') & (df['parent'] == us["id"])]
                task_blocks = [{
                    "task_id": task["id"],
                    "task_content": workitem_details(task)
                } for _, task in tasks.iterrows()]

                us_block = {
                    "user_story_id": us["id"],
                    "user_story_content": workitem_details(us),
                    "tasks": task_blocks
                }

                feature_block["user_stories"].append(us_block)
            epic_block["features"].append(feature_block)

        hierarchy.append(epic_block)

    return hierarchy

# ---------------------------
# Wiki Page Fetching (Optional)
# ---------------------------
def fetch_wiki_content(connection, wiki_url):
    parsed = urlparse(wiki_url)
    path_parts = parsed.path.strip('/').split('/')
    wiki_identifier = path_parts[4]
    page_id = path_parts[5]
    wiki_client = connection.clients.get_wiki_client()
    page = wiki_client.get_page_by_id(project=PROJECT_NAME, wiki_identifier=wiki_identifier, id=page_id, include_content=True)
    return page.page.content

# ---------------------------
# Main Function
# ---------------------------
def main():
    start_time = time.time()
    connection = authenticate()
    wit_client = connection.clients.get_work_item_tracking_client()

    fields_to_fetch = [
        "System.Id",
        "System.Title",
        "System.WorkItemType",
        "System.Description",
        "Microsoft.VSTS.Common.AcceptanceCriteria",
        "Custom.BusinessOutcomeHypothesis",
        "System.Parent"
    ]

    work_item_ids = fetch_work_item_ids(wit_client)
    print(f"Found {len(work_item_ids)} work items")

    all_items = fetch_work_items(wit_client, work_item_ids, fields_to_fetch)
    records = extract_work_items(all_items)

    df = pd.DataFrame(records)
    df.to_csv("extract_ado1.csv", index=False)

    hierarchy = convert_to_hierarchy(df)
    with open("nested_hierarchy.json", "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, indent=4, ensure_ascii=False)

    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds")

# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    main()
