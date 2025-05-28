import os
import re
import logging
import requests
from urllib.parse import urlparse, urlsplit, quote, parse_qs
from bs4 import BeautifulSoup

MAX_DEPTH = 5

def save_image(img_url, output_dir, base_url, pat_token):
    if img_url.startswith('/'):
        img_url = base_url + img_url
    headers = {"Authorization": f"Basic {requests.auth._basic_auth_str('', pat_token)}"}
    response = requests.get(img_url, headers=headers)
    if response.status_code == 200:
        filename = os.path.basename(urlparse(img_url).path)
        image_path = os.path.join(output_dir, "images", filename)
        with open(image_path, "wb") as f:
            f.write(response.content)
        logging.info(f"[✓] Image saved: {image_path}")
    else:
        logging.warning(f"[!] Failed to download image: {img_url} (status {response.status_code})")

def extract_and_save_images(content, soup, wiki_client, wiki_identifier, project_name, output_dir, pat_token):
    for img in soup.find_all('img', src=True):
        save_image(img['src'], output_dir, wiki_client.base_url, pat_token)

    try:
        wiki_data = wiki_client.get_wiki(project=project_name, wiki_identifier=wiki_identifier)
        repository_id = wiki_data.repository_id
        md_images = re.findall(r'!\[.*?\]\((.*?)\)', content)
        for rel_path in md_images:
            encoded_path = quote(rel_path)
            attachment_url = (
                f"{wiki_client.base_url}/{quote(project_name)}/_apis/git/repositories/{repository_id}/Items"
                f"?path={encoded_path}&download=false&resolveLfs=true&$format=octetStream"
            )
            resp = requests.get(attachment_url, auth=requests.auth.HTTPBasicAuth('', pat_token))
            if resp.status_code in (200, 203):
                filename = os.path.basename(encoded_path)
                image_path = os.path.join(output_dir, "images", filename)
                with open(image_path, "wb") as f:
                    f.write(resp.content)
                logging.info(f"[✓] Markdown image saved: {image_path}")
            else:
                logging.warning(f"[!] Skipped non-image or failed download: {rel_path} (status {resp.status_code})")
    except Exception as e:
        logging.error(f"[!] Markdown image handling failed: {e}")

def get_page_by_path(wiki_identifier, page_path, project_name, base_url, pat_token):
    encoded_path = quote(page_path, safe='')
    url = (
        f"{base_url}/{project_name}/_apis/wiki/wikis/{wiki_identifier}/pages"
        f"?path={encoded_path}&includeContent=true&api-version=7.0"
    )
    headers = {"Authorization": f"Basic {requests.auth._basic_auth_str('', pat_token)}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        logging.warning(f"[!] REST API failed for path {page_path} - status {response.status_code}")
        return None

def process_page(wiki_identifier, page_id, wiki_client, project_name, output_dir, base_url, pat_token):
    try:
        page = wiki_client.get_page_by_id(
            project=project_name,
            wiki_identifier=wiki_identifier,
            id=page_id,
            include_content=True
        )
    except Exception as e:
        logging.error(f"[!] Failed to fetch page {wiki_identifier}/{page_id}: {e}")
        return []

    content = page.page.content
    md_file = os.path.join(output_dir, f"page_{page_id}.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(content)
    logging.info(f"[✓] Wiki page saved: {md_file}")

    soup = BeautifulSoup(content, "html.parser")
    extract_and_save_images(content, soup, wiki_client, wiki_identifier, project_name, output_dir, pat_token)

    html_links = [a['href'] for a in soup.find_all('a', href=True)]
    markdown_links = re.findall(r'\[[^\]]*\]\((.*?)\)', content)
    wiki_links = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content)
    wiki_style_links = [
        f"/{project_name}/_wiki/wikis/{wiki_identifier}/pages?path=/{quote(title.strip())}"
        for title in wiki_links
    ]

    all_links = list(set(html_links + markdown_links + wiki_style_links))
    internal_links = []
    for link in all_links:
        if link.startswith('/'):
            if "pages?path=" in link:
                internal_links.append(link)
            else:
                internal_links.append(
                    f"/{project_name}/_wiki/wikis/{wiki_identifier}/pages?path={quote(link)}"
                )
    return internal_links

def crawl_wiki_page_recursive(wiki_client, wiki_identifier, page_id, project_name, base_url, pat_token, output_dir, visited, current_depth):
    if current_depth > MAX_DEPTH:
        return

    key = f"{wiki_identifier}_{page_id}"
    if key in visited:
        return
    visited.add(key)

    links = process_page(wiki_identifier, page_id, wiki_client, project_name, output_dir, base_url, pat_token)

    for link in links:
        try:
            if "/_wiki/wikis/" in link:
                parts = urlsplit(link)
                qs = parse_qs(parts.query)
                page_path = qs.get("path", [None])[0]
                if not page_path or page_path in ["/_TOC_", "/_Header", "/_Footer"]:
                    continue
                page_data = get_page_by_path(wiki_identifier, page_path, project_name, base_url, pat_token)
                if page_data and "id" in page_data:
                    next_page_id = page_data["id"]
                    crawl_wiki_page_recursive(
                        wiki_client, wiki_identifier, next_page_id, project_name,
                        base_url, pat_token, output_dir, visited, current_depth + 1
                    )
        except Exception as e:
            logging.warning(f"[!] Error processing link {link}: {e}")

def fetch_and_save_wiki_with_links_and_images(connection, wiki_url, project_name, pat_token, output_dir="wiki_pages"):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    parsed = urlparse(wiki_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    parts = parsed.path.strip('/').split('/')
    wiki_id = parts[4]
    page_id = parts[5]

    wiki_client = connection.clients.get_wiki_client()
    wiki_client.base_url = base_url

    visited = set()
    crawl_wiki_page_recursive(
        wiki_client=wiki_client,
        wiki_identifier=wiki_id,
        page_id=page_id,
        project_name=project_name,
        base_url=base_url,
        pat_token=pat_token,
        output_dir=output_dir,
        visited=visited,
        current_depth=1
    )
