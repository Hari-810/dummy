def fetch_and_save_wiki_with_links_and_images(connection, wiki_url, project_name=PROJECT_NAME, pat_token=PERSONAL_ACCESS_TOKEN, output_dir="wiki_pages"):
    import os, re
    from urllib.parse import urlparse, quote, urlsplit, parse_qs
    from bs4 import BeautifulSoup
    import requests

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    parsed = urlparse(wiki_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    path_parts = parsed.path.strip('/').split('/')
    initial_wiki_identifier = path_parts[4]
    initial_page_id = path_parts[5]

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
        try:
            page = wiki_client.get_page_by_id(
                project=PROJECT_NAME,
                wiki_identifier=wiki_identifier,
                id=page_id,
                include_content=True
            )
        except Exception as e:
            print(f"[!] Failed to fetch page {wiki_identifier}/{page_id}: {e}")
            return []
        
        # Save raw content to a file.
        content = page.page.content
        md_file = os.path.join(output_dir, f"page_{page_id}.md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[✓] Wiki page saved: {md_file}")

        soup = BeautifulSoup(content, "html.parser")
        # -- Save HTML-style images --
        for img in soup.find_all('img', src=True):
            save_image(img['src'])
        # -- Save Markdown-style images --
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
                resp = requests.get(attachment_url, auth=requests.auth.HTTPBasicAuth('', pat_token))
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

        # -- Extract HTML links --
        html_links = [a['href'] for a in soup.find_all('a', href=True)]
        # -- Extract Markdown links --
        markdown_links = re.findall(r'\[[^\]]*\]\((.*?)\)', content)
        # -- Extract Wiki-style links: [[PageName]] or [[PageName|Label]] --
        wiki_syntax_links = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content)
        wiki_style_internal_links = [
            f"/{project_name}/_wiki/wikis/{wiki_identifier}/pages?path=/{quote(title.strip())}"
            for title in wiki_syntax_links
        ]
        # -- Combine all extracted links --
        all_links = list(set(html_links + markdown_links + wiki_style_internal_links))
        # -- Filter internal wiki links: if a link starts with '/'
        internal_links = []
        for link in all_links:
            if link.startswith('/'):
                # If the link already contains "pages?path=", then assume it is in our normalized form.
                if "pages?path=" in link:
                    internal_links.append(link)
                else:
                    # Otherwise, treat it as a relative path and convert it.
                    normalized_link = f"/{project_name}/_wiki/wikis/{wiki_identifier}/pages?path={quote(link)}"
                    internal_links.append(normalized_link)
        return internal_links

    visited_pages = set()
    to_visit = [(initial_wiki_identifier, initial_page_id)]

    while to_visit:
        curr_wiki_id, curr_page_id = to_visit.pop()
        key = f"{curr_wiki_id}_{curr_page_id}"
        if key in visited_pages:
            continue
        visited_pages.add(key)

        links = process_page(curr_wiki_id, curr_page_id)

        for link in links:
            try:
                # Extract the 'path' parameter from the link.
                splitted = urlsplit(link)
                qs = parse_qs(splitted.query)
                page_path = qs.get("path", [None])[0]
                if page_path:
                    # Use the get_page_by_path API to retrieve the page using its relative path.
                    try:
                        page_obj = wiki_client.get_page_by_path(
                            project=PROJECT_NAME,
                            wiki_identifier=curr_wiki_id,
                            path=page_path,
                            include_content=True
                        )
                        new_page_id = page_obj.page.id
                        next_key = f"{curr_wiki_id}_{new_page_id}"
                        if next_key not in visited_pages:
                            to_visit.append((curr_wiki_id, new_page_id))
                    except Exception as e:
                        print(f"[!] Error resolving page path {page_path}: {e}")
                else:
                    print(f"[!] No 'path' found in link: {link}")
            except Exception as e:
                print(f"[!] Error processing link {link}: {e}")
                continue
