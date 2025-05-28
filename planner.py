import os
import re
import requests
from urllib.parse import quote, unquote, urlsplit, parse_qs
from bs4 import BeautifulSoup

class SimpleAzureWikiCrawler:
    SYSTEM_PATHS = {"/_TOC_", "/_Header", "/_Footer"}

    def __init__(self, wiki_client, project, pat_token, base_url, output_dir="wiki_pages", max_depth=5):
        self.wiki_client = wiki_client
        self.project = project
        self.pat_token = pat_token
        self.output_dir = output_dir
        self.max_depth = max_depth
        self.base_url = base_url.rstrip("/")
        self.visited = set()
        self.crawled_urls = set()
        self.headers = {
            "Authorization": f"Basic {requests.auth._basic_auth_str('', pat_token)}"
        }
        os.makedirs(self.output_dir, exist_ok=True)

    def save_crawled_urls(self, output_file="crawled_urls.txt"):
        with open(output_file, "w", encoding="utf-8") as f:
            for url in sorted(self.crawled_urls):
                f.write(url + "\n")
        print(f"[✓] Crawled URLs saved to: {output_file}")

    def save_page_content(self, content, page_id):
        safe_page_id = re.sub(r'[^\w\-_.]', '_', str(page_id))
        filepath = os.path.join(self.output_dir, f"page_{safe_page_id}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[✓] Saved page {safe_page_id} to {filepath}")

    def fetch_page_by_id(self, wiki_id, page_id):
        try:
            page = self.wiki_client.get_page_by_id(
                project=self.project, wiki_identifier=wiki_id,
                id=page_id, include_content=True
            )
            return page.page.content
        except Exception as e:
            print(f"[!] Failed to fetch page {wiki_id}/{page_id}: {e}")
            return None

    def fetch_page_by_path_rest(self, wiki_id, path):
        encoded_path = quote(path, safe="/")
        url = f"{self.base_url}/{self.project}/_apis/wiki/wikis/{wiki_id}/pages?path={encoded_path}&includeContent=true&api-version=7.0"
        response = requests.get(url, headers=self.headers)
        if response.status_code in (200, 203):
            json_data = response.json()
            return json_data.get("content")
        print(f"[!] Failed to fetch path {path}, status: {response.status_code}")
        return None

    def extract_links(self, content):
        soup = BeautifulSoup(content, "html.parser")
        html_links = [a['href'] for a in soup.find_all('a', href=True)]
        markdown_links = re.findall(r'\[[^\]]*\]\((.*?)\)', content)
        wiki_syntax_links = re.findall(r'\[\[([^\]|]+)', content)
        return set(html_links + markdown_links + wiki_syntax_links)

    def resolve_link(self, link, wiki_id):
        link = unquote(link.strip())

        if any(sys in link for sys in self.SYSTEM_PATHS):
            return None

        # Direct REST path-based URL
        if link.startswith("/_wiki/wikis/") and "pages?path=" in link:
            qs = parse_qs(urlsplit(link).query)
            path = qs.get("path", [None])[0]
            if path:
                return {"type": "path", "path": unquote(path)}

        # Full direct wiki link: /_wiki/wikis/{wikiId}/page/{pageId}/{title}
        if "/_wiki/wikis/" in link and "/page/" in link:
            match = re.search(r"/_wiki/wikis/([^/]+)/page/(\d+)", link)
            if match:
                return {"type": "id", "page_id": match.group(2)}

        # Custom project-specific wiki path (e.g., /Audit-AIML-Project-Wiki/pages/...)
        if link.startswith("/Audit-AIML-Project-Wiki"):
            page_path = link.split("/Audit-AIML-Project-Wiki")[-1]
            page_path = page_path.replace("/pages", "").strip("/")
            return {"type": "path", "path": page_path}

        return None

    def crawl(self, wiki_id, page_id_or_path, depth=1, is_path=False):
        if depth > self.max_depth:
            print(f"[i] Max depth {self.max_depth} reached.")
            return

        key = f"{wiki_id}_{page_id_or_path}"
        if key in self.visited:
            print(f"[i] Already visited: {key}")
            return

        self.visited.add(key)

        # Fetch page content
        if is_path:
            content = self.fetch_page_by_path_rest(wiki_id, page_id_or_path)
            page_id = page_id_or_path.replace("/", "_")
        else:
            content = self.fetch_page_by_id(wiki_id, page_id_or_path)
            page_id = page_id_or_path

        if not content:
            print(f"[!] No content found for: {page_id_or_path}")
            return

        self.save_page_content(content, page_id)
        self.crawled_urls.add(f"{wiki_id}_{page_id_or_path}")

        # Recurse through links
        links = self.extract_links(content)
        for link in links:
            info = self.resolve_link(link, wiki_id)
            if not info:
                continue
            if info["type"] == "path":
                self.crawl(wiki_id, info["path"], depth + 1, is_path=True)
            elif info["type"] == "id":
                self.crawl(wiki_id, info["page_id"], depth + 1, is_path=False)
