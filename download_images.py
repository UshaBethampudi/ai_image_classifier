import os, json, requests
from urllib.parse import urlsplit
from tqdm import tqdm

def ext_from_url(url):
    path = urlsplit(url).path
    ext = os.path.splitext(path)[1]
    return ext if ext else ".jpg"

def download_list(class_name, urls, out_root="data/raw"):
    os.makedirs(os.path.join(out_root, class_name), exist_ok=True)
    for i, url in enumerate(tqdm(urls, desc=class_name)):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                ext = ext_from_url(url)
                fname = f"{class_name}_{i+1:03d}{ext}"
                with open(os.path.join(out_root, class_name, fname), "wb") as f:
                    f.write(r.content)
        except Exception as e:
            print("skip", url, e)

if __name__ == "__main__":
    with open("urls.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    for cls, urls in data.items():
        download_list(cls, urls)
