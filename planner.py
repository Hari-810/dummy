
class SimpleAzureWikiCrawler:
    SYSTEM_PATHS = {"/_TOC_", "/_Header", "/_Footer"}

    def __init__(self, wiki_client, project, pat_token, base_url, output_dir="wiki_pages", max_depth=5):
        self.wiki_client = wiki_client
        self.project = project
        self.pat_token = pat_token
        self.output_dir = output_dir
        self.max_depth = max_depth
        self.base_url = base_url
        self.visited = set()
        self.crawled_urls = set()
        self.headers = {"Authorization": f"Basic {requests.auth._basic_auth_str('', pat_token)}"}

        os.makedirs(self.output_dir, exist_ok=True)
    def save_crawled_urls(self, output_file="crawled_urls.txt"):
        with open(output_file, "w", encoding="utf-8") as f:
            for url in sorted(self.crawled_urls):
                f.write(url + "\n")
        print(f"[✓] Crawled URLs saved to: {output_file}")

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
        if response.status_code in (200, 203):
            return response.text
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
        elif "/_wiki/wikis/" in link and link.count("/") >= 6:
            segments = urlsplit(link).path.strip('/').split('/')
            wiki_index = segments.index('_wiki')
            wiki_identifier = segments[wiki_index + 2]
            # Check if next segment after wiki_identifier is a digit (page ID)
            page_index = wiki_index + 3
            if len(segments) > page_index and segments[page_index].isdigit():
                page_id = segments[page_index]
            return {"type": "id", "page_id": page_id}
        elif link.startswith("/Audit-AIML-Project-Wiki"):
            # path = unquote(link)
            # splitted = urlsplit(link)
            # qs = parse_qs(splitted.path)
            # page_path = qs.get("path", [None])[0]
            # # if not page_path or page_path in ["/_TOC_", "/_Header", "/_Footer"]:
            # #     print(f"[!] Skipping system path or empty: {page_path}")
            # page_path_clean = unquote(page_path)
            # page_path_encoded = quote(page_path_clean, safe="/")
            prefix = "https://dev.azure.com/symphonyvsts/Audit%20AIML/_wiki/wikis/Audit-AIML.wiki?wikiVersion=GBwikiMaster&pagePath="
            modified_url = prefix + link
            print(f"[+] Modified URL")
            return {"type": "path", "path": modified_url}
        return None

    def crawl(self, wiki_id, page_id_or_path, depth=1, is_path=False):
        if depth > self.max_depth:
            return
        key = f"{wiki_id}_{page_id_or_path}"
        if key in self.visited:
            return
        self.visited.add(key)

        if is_path:
            content = self.fetch_page_by_path_rest(wiki_id, page_id_or_path)
            page_id = key.split("/")[-1]
            if not content:
                print(f"[!] Could not fetch {page_id_or_path}")
                return

        else:
            content = self.fetch_page_by_id(wiki_id, page_id_or_path)
            page_id = page_id_or_path

        if not content:
            return

        self.save_page_content(content, page_id)
        links = self.extract_links(content)
        print(links)
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

