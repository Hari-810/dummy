import os
import re
import requests
from urllib.parse import urlparse, unquote, quote
from bs4 import BeautifulSoup
from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication

# Constants
PROJECT_NAME = "Audit AIML"
WIKI_URL = 'https://dev.azure.com/symphonyvsts/Audit%20AIML/_wiki/wikis/Audit-AIML.wiki/178790/API-Testing-Using-VS-Code'
OUTPUT_DIR = "wiki_output"

# Replace with your actual PAT
PAT = "<your_personal_access_token>"
ENCODED_PAT = requests.auth._basic_auth_str("", PAT)

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

def fetch_and_save_wiki_with_links_and_images(connection, wiki_url, project_name, pat_token, output_dir="wiki_pages"):
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
        headers = {"Authorization": f"Basic {requests.auth._basic_auth_str('', PAT)}"}
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
                headers = {"Authorization": f"Basic {requests.auth._basic_auth_str('', pat_token)}"}
                resp = requests.get(attachment_url, headers=headers)
                if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("image/"):
                    filename = os.path.basename(encoded_path)
                    image_path = os.path.join(output_dir, "images", filename)
                    with open(image_path, "wb") as f:
                        f.write(resp.content)
                    print(f"[✓] Markdown image saved: {image_path}")
                else:
                    print(f"[!] Skipped non-image or failed download: {rel_path}")
            except Exception as e:
                print(f"[!] Error downloading markdown image {rel_path}: {e}")

        # Return internal wiki links
        links = [a['href'] for a in soup.find_all('a', href=True)]
        print("Links:", links)
        return links

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
    credentials = BasicAuthentication('', PAT)
    connection = Connection(base_url='https://dev.azure.com/symphonyvsts/', creds=credentials)

    fetch_and_save_wiki_with_links_and_images(
        connection=connection,
        wiki_url=WIKI_URL,
        project_name=PROJECT_NAME,
        pat_token=PAT,
        output_dir=OUTPUT_DIR
    )

if __name__ == "__main__":
    main()
