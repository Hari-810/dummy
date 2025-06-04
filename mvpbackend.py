from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication
import json
from bs4 import BeautifulSoup
from azure.devops.v7_1.work_item_tracking import Wiql
from azure.devops.v7_1.work_item_tracking import AttachmentReference
import pandas as pd
import time
start_time=time.time()
########## credential connection
personal_access_token = '1lObibwEdpFlnKvb3zVeiMUaxLIrWiW45WcaHj4mHlfmUIkHEC6nJQQJ99BEACAAAAAxJCnnAAASAZDO1u6x'# replace your code
organization_url = "https://dev.azure.com/symphonyvsts" # replace organization name 'https://dev.azure.com/symphonyvsts'
project_name ="Audit AIML" # replace project name 'Audit AIML' 

credentials = BasicAuthentication("", personal_access_token)
connection = Connection(base_url=organization_url, creds=credentials)

wit_client = connection.clients.get_work_item_tracking_client()


fields=wit_client.get_fields(project=project_name)
# Print all field names and references
for field in fields:
    print(f"{field.name} → {field.reference_name}")


core_client = connection.clients.get_core_client()  #  Core client to access projects 
projects = core_client.get_projects() # List all projects
print("Azure DevOps Projects:")
for project in projects:
    print(f"- {project.name}")




wiql_query = f"""
SELECT [System.Id]
FROM WorkItems
WHERE [System.TeamProject] = '{project_name}'
ORDER BY [System.Id] ASC
"""
wiql = Wiql(query=wiql_query)
query_result = wit_client.query_by_wiql(wiql=wiql)

work_item_ids = [item.id for item in query_result.work_items]
print(f"Found {len(work_item_ids)} work items")
print(work_item_ids)
# work_item_ids = work_item_ids[:1000]


def parse_html(html_content):
    soup = BeautifulSoup(html_content or "", "html.parser")
    text = soup.get_text(separator="\n")
    links = [a['href'] for a in soup.find_all('a', href=True)]
    images = [img['src'] for img in soup.find_all('img', src=True)]

    return {
        "text": text,
        "links": links,
        "images": images
    }

all_items = []
fields_to_fetch = [
    "System.Id",
    "System.Title",
    "System.WorkItemType",
    "System.Description",
    "Microsoft.VSTS.Common.AcceptanceCriteria",
    "Custom.BusinessOutcomeHypothesis",
    "System.Parent"
]

batch_size = 200
for i in range(0, len(work_item_ids), batch_size):
    batch_ids = work_item_ids[i:i+batch_size]
    work_items = wit_client.get_work_items(ids=batch_ids, fields=fields_to_fetch)
    all_items.append(work_items)


#all item data extraction 
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
                "parent":fields.get("System.Parent","")
            })
        except:
            pass

df=pd.DataFrame(records)
df.to_csv('extract_ado1.csv')
df.head()



def workitem_details(data):
    return {
    "epic_id": data.get("id",""),
    "title": data.get("title", ""),
    "description": data.get("description", ""),
    "work_item_type": data.get("work_item_type", ""),
    "description": (data.get("description", "")),
    "acceptance_criteria": (data.get("acceptance_criteria", "")),
    "business_outcome_hypothesis": data.get("business_outcome_hypothesis", "")
    }
def convert_into_hierarchy(df):
    epics = df[df['work_item_type'] == 'Epic']
    hierarchy = []
    for _, epic in epics.iterrows():
        epic_block = {
            "epic_id": epic["id"],
            "epic_content":workitem_details(epic),
            "features": []
        }
        features = df[(df['work_item_type'] == 'Feature') & (df['parent'] == epic["id"])]
        for _, feature in features.iterrows():
            feature_block = {
                "feature_id": feature["id"],
                "feature_content":workitem_details(feature),
                "user_stories": []
            }
            user_stories = df[(df['work_item_type'] == 'User Story') & (df['parent'] == feature["id"])]
            for _, us in user_stories.iterrows():
                tasks = df[(df['work_item_type'] == 'Task') & (df['parent'] == us["id"])]
                task_blocks = []
                for _, task in tasks.iterrows():
                    task_blocks.append({
                        "task_id": task["id"],
                        "task_content":workitem_details(task),
                    })
                us_block = {
                    "user_story_id": us["id"],
                    "user_story_content":workitem_details(us),
                    "tasks": task_blocks
                }
                feature_block["user_stories"].append(us_block)
            epic_block["features"].append(feature_block)
        hierarchy.append(epic_block)
    return hierarchy

result=convert_into_hierarchy(df)



with open("nested_hierarchy.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)


end_time=time.time()
total_timings=end_time-start_time
print(f"total timing {total_timings:.2f} seconds")




from urllib.parse import urlparse
wiki_url = 'https://dev.azure.com/symphonyvsts/Audit%20AIML/_wiki/wikis/Audit-AIML.wiki/178790/API-Testing-Using-VS-Code'
parsed = urlparse(wiki_url)
path_parts = parsed.path.strip('/').split('/')
wiki_identifier = path_parts[4] 
page_path_parts = path_parts[5:]
wiki_client = connection.clients.get_wiki_client()
i=wiki_client.get_page_by_id(project=project_name,wiki_identifier=wiki_identifier,id=page_path_parts[0],include_content=True)
content=i.page.content


    
