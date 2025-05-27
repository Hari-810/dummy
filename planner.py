import json
import time
from urllib.parse import urlparse
import pandas as pd
from bs4 import BeautifulSoup
from azure.devops.connection import Connection
from azure.devops.v7_1.work_item_tracking import Wiql
from msrest.authentication import BasicAuthentication

# Configuration
PERSONAL_ACCESS_TOKEN = '<YOUR_PAT_HERE>'
ORGANIZATION_URL = "https://dev.azure.com/symphonyvsts"
PROJECT_NAME = "Audit AIML"
WIKI_URL = 'https://dev.azure.com/symphonyvsts/Audit%20AIML/_wiki/wikis/Audit-AIML.wiki/178790/API-Testing-Using-VS-Code'

FIELDS_TO_FETCH = [
    "System.Id",
    "System.Title",
    "System.WorkItemType",
    "System.Description",
    "Microsoft.VSTS.Common.AcceptanceCriteria",
    "Custom.BusinessOutcomeHypothesis",
    "System.Parent"
]

BATCH_SIZE = 200


def authenticate():
    credentials = BasicAuthentication("", PERSONAL_ACCESS_TOKEN)
    connection = Connection(base_url=ORGANIZATION_URL, creds=credentials)
    return connection


def parse_html(html_content):
    soup = BeautifulSoup(html_content or "", "html.parser")
    text = soup.get_text(separator="\n")
    links = [a['href'] for a in soup.find_all('a', href=True)]
    images = [img['src'] for img in soup.find_all('img', src=True)]
    return {"text": text, "links": links, "images": images}


def fetch_work_item_data(connection):
    wit_client = connection.clients.get_work_item_tracking_client()

    wiql_query = f"""
    SELECT [System.Id]
    FROM WorkItems
    WHERE [System.TeamProject] = '{PROJECT_NAME}'
    ORDER BY [System.Id] ASC
    """
    wiql = Wiql(query=wiql_query)
    query_result = wit_client.query_by_wiql(wiql=wiql)
    work_item_ids = [item.id for item in query_result.work_items]
    print(f"Found {len(work_item_ids)} work items")

    all_items = []
    for i in range(0, len(work_item_ids), BATCH_SIZE):
        batch_ids = work_item_ids[i:i + BATCH_SIZE]
        work_items = wit_client.get_work_items(ids=batch_ids, fields=FIELDS_TO_FETCH)
        all_items.append(work_items)

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
    return pd.DataFrame(records)


def workitem_details(data):
    return {
        "epic_id": data.get("id", ""),
        "title": data.get("title", ""),
        "description": data.get("description", ""),
        "work_item_type": data.get("work_item_type", ""),
        "acceptance_criteria": data.get("acceptance_criteria", ""),
        "business_outcome_hypothesis": data.get("business_outcome_hypothesis", "")
    }


def convert_into_hierarchy(df):
    epics = df[df['work_item_type'] == 'Epic']
    hierarchy = []
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


def fetch_and_save_wiki(connection, wiki_url, output_file="wiki_page_content.md"):
    parsed = urlparse(wiki_url)
    path_parts = parsed.path.strip('/').split('/')
    wiki_identifier = path_parts[4]
    page_id = path_parts[5]

    wiki_client = connection.clients.get_wiki_client()
    page = wiki_client.get_page_by_id(
        project=PROJECT_NAME,
        wiki_identifier=wiki_identifier,
        id=page_id,
        include_content=True
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(page.page.content)
    print(f"Wiki content saved to {output_file}")


def main():
    start_time = time.time()

    connection = authenticate()
    all_items = fetch_work_item_data(connection)
    df = extract_work_items(all_items)
    df.to_csv('extract_ado1.csv', index=False)
    print("Work item data saved to extract_ado1.csv")

    hierarchy = convert_into_hierarchy(df)
    with open("nested_hierarchy.json", "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, indent=4, ensure_ascii=False)
    print("Hierarchy saved to nested_hierarchy.json")

    fetch_and_save_wiki(connection, WIKI_URL)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
