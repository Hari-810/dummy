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
                    f"https://dev.azure.com/symphonyvsts/"
                    f"{quote(self.project_name)}/_apis/git/repositories/{repository_id}/Items"
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

        html_links = [a['href'] for a in soup.find_all('a', href=True)]
        markdown_links = re.findall(r'\[[^\]]*\]\((.*?)\)', content)
        wiki_syntax_links = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content)
        wiki_style_internal_links = [
            f"/{self.project_name}/_wiki/wikis/{wiki_identifier}/pages?path=/{quote(title.strip())}"
            for title in wiki_syntax_links
        ]

        all_links = list(set(html_links + markdown_links + wiki_style_internal_links))
        self.all_urls.update(all_links)

        internal_links = []
        for link in all_links:
            if link.startswith('/'):
                if "pages?path=" in link:
                    internal_links.append(link)
                else:
                    normalized_link = f"/{self.project_name}/_wiki/wikis/{wiki_identifier}/pages?path={quote(link)}"
                    internal_links.append(normalized_link)
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
                    if "/Audit-AIML-Project-Wiki/API-Platform/" in link:
                        print(f"[+] Matched API-Platform URL: {link}")
                        splitted = urlsplit(link)
                        qs = parse_qs(splitted.query)
                        page_path = qs.get("path", [None])[0]

                        if not page_path or page_path in ["/_TOC_", "/_Header", "/_Footer"]:
                            print(f"[!] Skipping system path or empty: {page_path}")
                            continue

                        page_path_clean = unquote(page_path)
                        page_path_encoded = quote(page_path_clean, safe="/")

                        prefix = "https://dev.azure.com/symphonyvsts/Audit%20AIML/_wiki/wikis/Audit-AIML.wiki?wikiVersion=GBwikiMaster&pagePath="
                        modified_url = prefix + page_path_encoded
                        print(f"[+] Modified URL: {modified_url}")

                        page_data = self.get_page_by_path_rest(wiki_identifier, page_path_clean)
                        if page_data:
                            new_page_id = page_data.get("id")
                            if new_page_id:
                                crawl_recursive(wiki_identifier, new_page_id, depth + 1)
                        else:
                            print(f"[!] REST API failed for path {page_path} - trying direct fetch")
                            response = requests.get(modified_url, headers=self.headers)
                            if response.status_code == 200:
                                print(f"[✓] Fallback fetch success: {modified_url}")
                                save_html_or_markdown(modified_url, response.text)
                            else:
                                print(f"[!] Fallback fetch failed. Status: {response.status_code}")
                        continue

                    elif "/_wiki/wikis/" in link:
                        if "pages?path=" in link:
                            splitted = urlsplit(link)
                            qs = parse_qs(splitted.query)
                            page_path = qs.get("path", [None])[0]
                            if not page_path or page_path in ["/_TOC_", "/_Header", "/_Footer"]:
                                continue
                            page_data = self.get_page_by_path_rest(wiki_identifier, page_path)
                            if page_data:
                                new_page_id = page_data.get("id")
                                if new_page_id:
                                    crawl_recursive(wiki_identifier, new_page_id, depth + 1)
                        else:
                            segments = urlsplit(link).path.strip('/').split('/')
                            try:
                                wiki_index = segments.index('_wiki')
                                wiki_identifier = segments[wiki_index + 2]
                                if 'pages' in segments:
                                    page_index = segments.index('pages')
                                    page_id = segments[page_index + 1]
                                    if page_id.isdigit():
                                        crawl_recursive(wiki_identifier, page_id, depth + 1)
                            except (ValueError, IndexError):
                                continue
                except Exception as e:
                    print(f"[!] Error processing link {link}: {e}")
                    continue

        crawl_recursive(initial_wiki_identifier, initial_page_id, 1)

        url_file = os.path.join(self.output_dir, "crawled_urls.txt")
        with open(url_file, "w", encoding="utf-8") as f:
            for url in sorted(self.all_urls):
                f.write(url + "\n")
        print(f"[✓] All URLs saved to: {url_file}")
