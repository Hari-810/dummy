import os
import re
import requests
from urllib.parse import urlparse, quote, unquote, urlsplit, parse_qs
from bs4 import BeautifulSoup

class AzureWikiCrawler:
    def __init__(self, connection, wiki_url, project_name, pat_token, output_dir="wiki_pages", max_depth=5):
        self.connection = connection
        self.wiki_url = wiki_url
        self.project_name = project_name
        self.pat_token = pat_token
        self.output_dir = output_dir
        self.max_depth = max_depth

        self.visited_pages = set()
        self.all_urls = set()

        self.parsed_url = urlparse(wiki_url)
        self.base_url = f"{self.parsed_url.scheme}://{self.parsed_url.netloc}"

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)

        self.wiki_client = self.connection.clients.get_wiki_client()
        self.headers = {"Authorization": f"Basic {requests.auth._basic_auth_str('', self.pat_token)}"}

    def save_image(self, img_url):
        if img_url.startswith('/'):
            img_url = self.base_url + img_url
        response = requests.get(img_url, headers=self.headers)
        if response.status_code == 200:
            filename = os.path.basename(urlparse(img_url).path)
            image_path = os.path.join(self.output_dir, "images", filename)
            with open(image_path, "wb") as f:
                f.write(response.content)
            print(f"[✓] Image saved: {image_path}")
        else:
            print(f"[!] Failed to download image: {img_url} (status {response.status_code})")

    def get_page_by_path_rest(self, wiki_identifier, page_path):
        encoded_path = quote(page_path, safe='')
        url = (
            f"{self.base_url}/{self.project_name}/_apis/wiki/wikis/{wiki_identifier}/pages"
            f"?path={encoded_path}&includeContent=true&api-version=7.0"
        )
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[!] REST API failed for path {page_path} - status {response.status_code}")
            return None

    def process_page(self, wiki_identifier, page_id):
        try:
            page = self.wiki_client.get_page_by_id(
                project=self.project_name,
                wiki_identifier=wiki_identifier,
                id=page_id,
                include_content=True
            )
        except Exception as e:
            print(f"[!] Failed to fetch page {wiki_identifier}/{page_id}: {e}")
            return []

        content = page.page.content
        md_file = os.path.join(self.output_dir, f"page_{page_id}.md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[✓] Wiki page saved: {md_file}")

        soup = BeautifulSoup(content, "html.parser")
        for img in soup.find_all('img', src=True):
            self.save_image(img['src'])

        try:
            wiki_data = self.wiki_client.get_wiki(project=self.project_name, wiki_identifier=wiki_identifier)
            repository_id = wiki_data.repository_id
            md_attachments = re.findall(r'!\[.*?\]\((.*?)\)', content)
            for rel_path in md_attachments:
                encoded_path = quote(rel_path)
                attachment_url = (
                    f"{self.base_url}/{quote(self.project_name)}/_apis/git/repositories/{repository_id}/Items"
                    f"?path={encoded_path}&download=false&resolveLfs=true&$format=octetStream"
                )
                resp = requests.get(attachment_url, auth=requests.auth.HTTPBasicAuth('', self.pat_token))
                if resp.status_code in (200, 203):
                    filename = os.path.basename(encoded_path)
                    image_path = os.path.join(self.output_dir, "images", filename)
                    with open(image_path, "wb") as f:
                        f.write(resp.content)
                    print(f"[✓] Markdown image saved: {image_path}")
                else:
                    print(f"[!] Skipped non-image or failed download: {rel_path} (status {resp.status_code})")
        except Exception as e:
            print(f"[!] Markdown image handling failed: {e}")

        # Extract all kinds of links
        html_links = [a['href'] for a in soup.find_all('a', href=True)]
        markdown_links = re.findall(r'\[[^\]]*\]\((.*?)\)', content)
        wiki_syntax_links = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content)
        direct_wiki_links = re.findall(r'/wiki/wikis/[^/]+/\d+/', content)

        wiki_style_internal_links = [
            f"/{self.project_name}/_wiki/wikis/{wiki_identifier}/pages?path=/{quote(title.strip())}"
            for title in wiki_syntax_links
        ]

        all_links = list(set(html_links + markdown_links + wiki_style_internal_links + direct_wiki_links))
        self.all_urls.update(all_links)

        internal_links = []
        for link in all_links:
            if "sharepoint.com" in link or "onedrive" in link:
                continue  # skip non-AzureDevOps links
            if "pages?path=" in link or re.search(r'/wiki/wikis/[^/]+/\d+/', link):
                internal_links.append(link)

        return internal_links

    def crawl(self):
        path_parts = self.parsed_url.path.strip('/').split('/')
        initial_wiki_identifier = path_parts[4]
        initial_page_id = path_parts[5]

        def crawl_recursive(wiki_identifier, page_id, depth):
            if depth > self.max_depth:
                return
            key = f"{wiki_identifier}_{page_id}"
            if key in self.visited_pages:
                return
            self.visited_pages.add(key)

            links = self.process_page(wiki_identifier, page_id)
            for link in links:
                try:
                    if "pages?path=" in link:
                        qs = parse_qs(urlsplit(link).query)
                        page_path = qs.get("path", [None])[0]
                        if not page_path or page_path in ["/_TOC_", "/_Header", "/_Footer"]:
                            continue
                        page_data = self.get_page_by_path_rest(wiki_identifier, page_path)
                        if page_data:
                            new_page_id = page_data.get("id")
                            if new_page_id:
                                crawl_recursive(wiki_identifier, new_page_id, depth + 1)
                    else:
                        match = re.search(r'/wiki/wikis/([^/]+)/(\d+)', link)
                        if match:
                            linked_wiki_identifier = match.group(1)
                            linked_page_id = match.group(2)
                            crawl_recursive(linked_wiki_identifier, linked_page_id, depth + 1)
                except Exception as e:
                    print(f"[!] Error processing link {link}: {e}")
                    continue

        crawl_recursive(initial_wiki_identifier, initial_page_id, 1)

        with open(os.path.join(self.output_dir, "crawled_urls.txt"), "w", encoding="utf-8") as f:
            for url in sorted(self.all_urls):
                f.write(url + "\n")
        print(f"[✓] All URLs saved to: {self.output_dir}/crawled_urls.txt")
