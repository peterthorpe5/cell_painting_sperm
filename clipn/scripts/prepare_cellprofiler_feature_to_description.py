#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

print("this doesnt work. Was harder than I thought")

BASE = "https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.8/modules/"

def get_module_pages():
    index_url = BASE + "index.html"
    r = requests.get(index_url)
    soup = BeautifulSoup(r.content, features="lxml")
    module_urls = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.endswith('.html') and not href.startswith('http'):
            module_urls.append(urljoin(BASE, href))
    return module_urls

def parse_features(url):
    """
    Extract features and descriptions from <li> elements and tables.
    """
    r = requests.get(url)
    soup = BeautifulSoup(r.content, features="lxml")
    features = []
    # Look for bullet lists
    for ul in soup.find_all('ul'):
        for li in ul.find_all('li'):
            txt = li.get_text(separator=" ", strip=True)
            if ':' in txt:
                # Split as "Feature: Description"
                feat, desc = txt.split(':', 1)
                features.append((feat.strip(), desc.strip()))
    # Also try tables (feature in first column, description in second)
    for table in soup.find_all('table'):
        for row in table.find_all('tr'):
            cols = row.find_all(['td', 'th'])
            if len(cols) >= 2:
                feature = cols[0].get_text(strip=True)
                desc = cols[1].get_text(strip=True)
                # Skip headers
                if feature.lower() in ['feature', 'measurement']:
                    continue
                features.append((feature, desc))
    return features

module_urls = get_module_pages()
all_features = []
for url in module_urls:
    feats = parse_features(url)
    all_features.extend(feats)

with open("cellprofiler_feature_dictionary.tsv", "w") as out:
    out.write("feature\tdescription\n")
    for feat, desc in all_features:
        out.write(f"{feat}\t{desc}\n")

print(f"Wrote {len(all_features)} features to cellprofiler_feature_dictionary.tsv")
