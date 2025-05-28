from urllib.parse import urlsplit, parse_qs, unquote, quote

class SimpleAzureWikiCrawler:
    SYSTEM_PATHS = {"/_TOC_", "/_Header", "/_Footer"}

    def __init__(self, wiki_client, project, pat_token, base_url, output_dir="wiki_pages", max_depth=5):
        self.wiki_client = wiki_client
        self.project = project
        self.pat_token = pat_token
        self.output_dir = output_dir
        self.max_depth = max_depth
        self.visited = set()
        self.headers = {"Authorization": f"Basic {requests.auth._basic_auth_str('', pat_token)}"}

        os.makedirs(self.output_dir, exist_ok=True)

    def save_page_content(self, content, page_id):
        filepath = os.path.join(self.output_dir, f"page_{page_id}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[✓] Saved page {page_id} to {filepath}")

    def fetch_page_by_id(self, wiki_id, page_id):
        try:
            page = self.wiki_client.get_page_by_id(project=self.project, wiki_identifier=wiki_id, id=page_id, include_content=True)
            return page.page.content
        except Exception as e:
            print(f"[!] Failed to fetch page {wiki_id}/{page_id}: {e}")
            return None

    def fetch_page_by_path_rest(self, wiki_id, path):
        encoded_path = quote(path, safe='')
        url = f"{self.base_url}/{self.project}/_apis/wiki/wikis/{wiki_id}/pages?path={encoded_path}&includeContent=true&api-version=7.0"
        response = requests.get(url, headers=self.headers)
        if response.ok:
            return response.json()
        return None

    def extract_links(self, content):
        soup = BeautifulSoup(content, "html.parser")
        html_links = [a['href'] for a in soup.find_all('a', href=True)]
        markdown_links = re.findall(r'\[[^\]]*\]\((.*?)\)', content)
        wiki_syntax_links = re.findall(r'\[\[([^\]|]+)', content)
        return set(html_links + markdown_links + wiki_syntax_links)

    def resolve_link(self, link, wiki_id):
        if link.startswith("/_wiki/wikis/") and "pages?path=" in link:
            qs = parse_qs(urlsplit(link).query)
            path = qs.get("path", [None])[0]
            return {"type": "path", "path": unquote(path)}
        elif link.startswith("/_wiki/wikis/") and link.count("/") >= 6:
            parts = link.strip("/").split("/")
            page_id = parts[5] if parts[5].isdigit() else None
            return {"type": "id", "page_id": page_id}
        elif link.startswith("/Audit-AIML-Project-Wiki"):
            path = unquote(link)
            return {"type": "path", "path": path}
        return None

    def crawl(self, wiki_id, page_id_or_path, depth=1, is_path=False):
        if depth > self.max_depth:
            return
        key = f"{wiki_id}_{page_id_or_path}"
        if key in self.visited:
            return
        self.visited.add(key)

        if is_path:
            data = self.fetch_page_by_path_rest(wiki_id, page_id_or_path)
            if not data:
                print(f"[!] Could not fetch {page_id_or_path}")
                return
            page_id = data.get("id")
            content = data.get("content", "")
        else:
            content = self.fetch_page_by_id(wiki_id, page_id_or_path)
            page_id = page_id_or_path

        if not content:
            return

        self.save_page_content(content, page_id)
        links = self.extract_links(content)

        for link in links:
            if any(sys in link for sys in self.SYSTEM_PATHS):
                continue
            info = self.resolve_link(link, wiki_id)
            if not info:
                continue
            if info["type"] == "path":
                self.crawl(wiki_id, info["path"], depth + 1, is_path=True)
            elif info["type"] == "id":
                self.crawl(wiki_id, info["page_id"], depth + 1, is_path=False)





crawler = SimpleAzureWikiCrawler(wiki_client, "Audit AIML", "your_PAT", "https://dev.azure.com/symphonyvsts")
crawler.crawl(wiki_id="Audit-AIML.wiki", page_id_or_path="111555", is_path=False)

