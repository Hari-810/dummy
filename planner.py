import os
import re
import requests
import logging
from urllib.parse import quote, urlparse, parse_qs
from bs4 import BeautifulSoup
from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication
from tenacity import retry, stop_after_attempt, wait_exponential

class AzureWikiCrawler:
    def __init__(self, organization_url, project_name, personal_access_token, repository_id, output_dir, max_depth=3):
        self.organization_url = organization_url.rstrip("/")
        self.project_name = project_name
        self.pat = personal_access_token
        self.repository_id = repository_id
        self.output_dir = output_dir
        self.max_depth = max_depth

        self.visited_pages = set()
        self.all_urls = set()

        credentials = BasicAuthentication("", self.pat)
        self.connection = Connection(base_url=self.organization_url, creds=credentials)
        self.git_client = self.connection.clients.get_git_client()

        self.headers = {
            "Authorization": f"Basic {self.pat.encode('utf-8').decode('latin1')}",
        }

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        logging.basicConfig(level=logging.INFO)

    def normalize_internal_link(self, link, wiki_identifier):
        if "pages?path=" not in link:
            path = quote(link.strip("/"), safe="/")
            return f"/{self.project_name}/_wiki/wikis/{wiki_identifier}/pages?path=/{path}"
        return link

    def safe_filename(self, filename):
        return re.sub(r'[\\/*?:"<>|]', "_", filename)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def download_with_retry(self, url):
        return requests.get(url, headers=self.headers)

    def save_image(self, img_url, save_dir):
        try:
            response = self.download_with_retry(img_url)
            if response.status_code == 200:
                filename = self.safe_filename(os.path.basename(urlparse(img_url).path))
                image_path = os.path.join(save_dir, filename)
                with open(image_path, "wb") as f:
                    f.write(response.content)
                logging.info(f"[✓] Saved: {image_path}")
        except Exception as e:
            logging.warning(f"[!] Failed to download: {img_url} | Reason: {str(e)}")

    def extract_and_save_images(self, page_content, save_dir):
        markdown_image_pattern = r"!\[.*?\]\((.*?)\)"
        image_urls = re.findall(markdown_image_pattern, page_content)
        for img_url in image_urls:
            if not img_url.startswith("http"):
                img_url = f"{self.organization_url}/{self.project_name}/_apis/git/repositories/{self.repository_id}/items?path={quote(img_url)}"
            self.save_image(img_url, save_dir)

        soup = BeautifulSoup(page_content, "html.parser")
        for img_tag in soup.find_all("img"):
            img_url = img_tag.get("src")
            if img_url and not img_url.startswith("http"):
                img_url = f"{self.organization_url}/{img_url.lstrip('/')}"
            self.save_image(img_url, save_dir)

    def extract_internal_links(self, page_content, wiki_identifier):
        internal_links = set()

        soup = BeautifulSoup(page_content, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag['href']
            if wiki_identifier in href:
                internal_links.add(self.normalize_internal_link(href, wiki_identifier))

        wiki_link_pattern = r"\[\[(.*?)\]\]"
        for match in re.findall(wiki_link_pattern, page_content):
            title = match.split("|")[0].strip()
            internal_links.add(self.normalize_internal_link(title, wiki_identifier))

        markdown_link_pattern = r"\[.*?\]\((.*?)\)"
        for match in re.findall(markdown_link_pattern, page_content):
            if wiki_identifier in match:
                internal_links.add(self.normalize_internal_link(match, wiki_identifier))

        return internal_links

    def fetch_page_content(self, url):
        response = requests.get(url, headers=self.headers)
        return response.text if response.status_code == 200 else None

    def crawl_page(self, url, depth, wiki_identifier):
        if depth > self.max_depth or url in self.visited_pages:
            return

        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        path = query_params.get("path", [None])[0]
        if path and path.lower() in ["/_toc_", "/_header", "/_footer"]:
            return

        self.visited_pages.add(url)
        response = requests.get(url, headers=self.headers)
        page_content = response.text if response.status_code == 200 else self.fetch_page_content(url)

        if not page_content:
            logging.warning(f"[!] Skipped: {url}")
            return

        safe_path = self.safe_filename(path.strip("/") if path else "root")
        save_path = os.path.join(self.output_dir, f"{safe_path}.html")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(page_content)

        self.extract_and_save_images(page_content, self.output_dir)

        internal_links = self.extract_internal_links(page_content, wiki_identifier)
        for link in internal_links:
            self.all_urls.add(link)
            self.crawl_page(link, depth + 1, wiki_identifier)

    def crawl(self, start_url):
        parsed = urlparse(start_url)
        wiki_identifier = parsed.path.split("/")[4]  # /project/_wiki/wikis/{wiki_id}/pages?path=...
        self.crawl_page(start_url, 0, wiki_identifier)

        url_file = os.path.join(self.output_dir, "all_urls.txt")
        with open(url_file, "w", encoding="utf-8") as f:
            for url in sorted(self.all_urls):
                f.write(url + "\n")
        logging.info(f"[✓] All extracted URLs saved to: {url_file}")



start_url = "https://dev.azure.com/your-org/your-project/_wiki/wikis/your-wiki-id/pages?path=/Home"
crawler.crawl(start_url)
