import json
import time
from urllib.parse import urlparse
import pandas as pd
from bs4 import BeautifulSoup
from azure.devops.connection import Connection
from azure.devops.v7_1.work_item_tracking import Wiql
from msrest.authentication import BasicAuthentication
import os
import requests
from urllib.parse import urlparse, unquote
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse, unquote, quote
# Configuration
# PERSONAL_ACCESS_TOKEN = '1lObibwEdpFlnKvb3zVeiMUaxLIrWiW45WcaHj4mHlfmUIkHEC6nJQQJ99BEACAAAAAxJCnnAAASAZDO1u6x'
PERSONAL_ACCESS_TOKEN = "EyR00Un9r3CDSaQeLIkEMMw0Aa4eU1Z1l6gMIVWjiOK0XXnRdbAyJQQJ99BEACAAAAAxJCnnAAASAZDO3IdD"
ORGANIZATION_URL = "https://dev.azure.com/symphonyvsts"
PROJECT_NAME = "Audit AIML"
WIKI_URL = 'https://dev.azure.com/symphonyvsts/Audit%20AIML/_wiki/wikis/Audit-AIML.wiki/148187/Smart-Review-plan'

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

def save_image_from_url(img_url, output_dir, pat_token):
    headers = {
        "Authorization": f"Basic {pat_token}"
    }

    response = requests.get(img_url, headers=headers)
    if response.status_code == 200:
        filename = os.path.basename(urlparse(img_url).path)
        image_path = os.path.join(output_dir, "images", filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        with open(image_path, "wb") as f:
            f.write(response.content)
        print(f"[✓] Image saved: {image_path}")
    else:
        print(f"[!] Failed to download image: {img_url} (status {response.status_code})")

def fetch_and_save_wiki_with_links_and_images(connection, wiki_url, project_name= PROJECT_NAME, pat_token=PERSONAL_ACCESS_TOKEN, output_dir="wiki_pages"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    parsed = urlparse(wiki_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    path_parts = parsed.path.strip('/').split('/')
    wiki_identifier = path_parts[4]
    page_id = path_parts[5]

    wiki_client = connection.clients.get_wiki_client()

    def save_image(img_url):
        if img_url.startswith('/'):
            img_url = base_url + img_url
        headers = {"Authorization": f"Basic {requests.auth._basic_auth_str('', pat_token)}"}
        response = requests.get(img_url, headers=headers)
        if response.status_code == 200:
            filename = os.path.basename(urlparse(img_url).path)
            image_path = os.path.join(output_dir, "images", filename)
            with open(image_path, "wb") as f:
                f.write(response.content)
            print(f"[✓] Image saved: {image_path}")
        else:
            print(f"[!] Failed to download image: {img_url} (status {response.status_code})")

    def process_page(wiki_identifier, page_id):
        page = wiki_client.get_page_by_id(
            project=PROJECT_NAME,
            wiki_identifier=wiki_identifier,
            id=page_id,
            include_content=True
        )
        content = page.page.content
        md_file = os.path.join(output_dir, f"page_{page_id}.md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[✓] Wiki page saved: {md_file}")

        soup = BeautifulSoup(content, "html.parser")

        # Save HTML <img> tag images
        for img in soup.find_all('img', src=True):
            save_image(img['src'])

        # Save markdown-style images (![alt](path))
        wiki_data = wiki_client.get_wiki(project=PROJECT_NAME, wiki_identifier=wiki_identifier)
        repository_id = wiki_data.repository_id
        md_attachments = re.findall(r'!\[.*?\]\((.*?)\)', content)
        for rel_path in md_attachments:
            encoded_path = quote(rel_path)
            
            attachment_url = (
                f"https://dev.azure.com/symphonyvsts/"
                f"{quote(PROJECT_NAME)}/_apis/git/repositories/{repository_id}/Items"
                f"?path={encoded_path}&download=false&resolveLfs=true&$format=octetStream"
            )
            try:
                import requests
                from requests.auth import HTTPBasicAuth
                # headers = {"Authorization": f"Basic {requests.auth._basic_auth_str('', pat_token)}"}
                # resp = requests.get(attachment_url, headers=headers)
                resp = requests.get(attachment_url, auth=HTTPBasicAuth('', pat_token))
                if resp.status_code in (200, 203):
                    filename = os.path.basename(encoded_path)
                    image_path = os.path.join(output_dir, "images", filename)
                    with open(image_path, "wb") as f:
                        f.write(resp.content)
                    print(f"[✓] Markdown image saved: {image_path}")
                else:
                    print(f"[!] Skipped non-image or failed download: {rel_path} (status {resp.status_code})")
            except Exception as e:
                print(f"[!] Error downloading markdown image {rel_path}: {e}")

        # Extract links using BeautifulSoup (HTML links)
        html_links = [a['href'] for a in soup.find_all('a', href=True)]

        # Extract links using regex for markdown format: [text](link)
        markdown_links = re.findall(r'\[[^\]]*\]\((.*?)\)', content)
        print("markdown_links :", markdown_links)
        # Combine and filter only internal wiki links
        all_links = html_links + markdown_links

        # Capture links pointing to other wiki pages
        wiki_links = [
            link for link in all_links
            if ('/wiki/' in link or '_wiki' in link)  # wiki structure match
        ]

        # Remove duplicates
        internal_links = list(set(wiki_links))

        print("Links:", internal_links)
        return internal_links

    visited_pages = set()
    to_visit = [(wiki_identifier, page_id)]

    while to_visit:
        curr_wiki_id, curr_page_id = to_visit.pop()
        key = f"{curr_wiki_id}_{curr_page_id}"
        if key in visited_pages:
            continue
        visited_pages.add(key)

        links = process_page(curr_wiki_id, curr_page_id)

        for link in links:
            if link.startswith('/'):
                try:
                    link_parts = link.strip('/').split('/')
                    if 'wiki' in link_parts and len(link_parts) >= 6:
                        new_wiki_id = link_parts[4]
                        new_page_id = link_parts[5]
                        to_visit.append((new_wiki_id, new_page_id))
                except:
                    continue


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
    # fetch_and_save_wiki(connection, WIKI_URL)
    fetch_and_save_wiki_with_links_and_images(connection, WIKI_URL)
    print("wiki ")
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
