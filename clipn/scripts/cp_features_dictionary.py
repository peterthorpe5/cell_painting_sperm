
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified CellProfiler feature dictionary tool (scraper + offline regex expander).

Modes
-----
1) Scrape manual pages (online or offline HTML) to build feature -> description TSV:
    python cp_features_dictionary.py \

        --mode scrape \

        --out_tsv cp_features.tsv \

        --include_generic_prefixes \

        --manual_version 4.2.8
   OR
    python cp_features_dictionary.py \

        --mode scrape \

        --html_dir /path/to/CellProfiler-4.2.8/modules \

        --out_tsv cp_features.tsv \

        --include_generic_prefixes

2) Expand from your dataset header using regex rules (robust, offline, recommended):
    python cp_features_dictionary.py \

        --mode expand \

        --scan_file profiles.tsv \

        --delimiter "\t" \

        --out_features cp_features.tsv

3) Always available: write the pattern dictionary (regex -> template) to TSV:
    python cp_features_dictionary.py --out_patterns cp_patterns.tsv

Behaviour
---------
- Keyword-only arguments only. All outputs are TSV (never CSV).
- British English spelling in docstrings and descriptions.
- If both scrape and expand arguments are supplied, --mode decides which logic to run.
- The scraper is resilient but can be blocked by network/S3 rules; the expander is
  deterministic and works fully offline against your actual column names.

"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import html
import logging
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Pattern, Tuple

# Optional imports (used conditionally)
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

try:
    import requests
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    requests = None
    BeautifulSoup = None  # type: ignore

from urllib.parse import urljoin, urlparse

DEFAULT_MANUAL_VERSION = "4.2.8"
DEFAULT_BASE = "https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-{ver}/"


@dataclasses.dataclass
class FeatureRow:
    """Container for a single feature description record."""
    feature: str
    description: str
    module: str
    source_url: str


# ------------------------------------------------------------------------------------------
# Pattern/Regex-based offline expander
# ------------------------------------------------------------------------------------------

@dataclasses.dataclass
class PatternRule:
    """A single regex rule mapping features to a description template."""
    name: str
    regex: Pattern[str]
    template: str
    family: str


def humanise_stat(stat: str) -> str:
    """Map common intensity/stat tokens to a friendly phrase."""
    m = {
        "MeanIntensity": "Mean intensity within the object",
        "MedianIntensity": "Median intensity within the object",
        "MinIntensity": "Minimum pixel intensity within the object",
        "MaxIntensity": "Maximum pixel intensity within the object",
        "IntegratedIntensity": "Integrated (summed) intensity within the object",
        "StdIntensity": "Standard deviation of intensities within the object",
        "MADIntensity": "Median absolute deviation of intensities within the object",
        "LowerQuartileIntensity": "Lower quartile intensity within the object",
        "UpperQuartileIntensity": "Upper quartile intensity within the object",
        "TotalIntensity": "Total intensity measure for the object",
        "MeanIntensityEdge": "Mean intensity along the object boundary",
        "StdIntensityEdge": "Standard deviation of intensity along the object boundary",
        "MassDisplacement": "Distance between intensity centre-of-mass and geometric centre",
    }
    return m.get(stat, stat.replace("_", " "))


def humanise_texture(feat: str) -> str:
    """Map Haralick texture tokens to friendly names."""
    m = {
        "AngularSecondMoment": "Energy (uniformity of grey-levels)",
        "Contrast": "Contrast (local intensity variations)",
        "Correlation": "Correlation (linear dependency of grey-levels)",
        "Variance": "Variance (grey-level variance)",
        "InverseDifferenceMoment": "Inverse Difference Moment (homogeneity)",
        "SumAverage": "Sum average (co-occurrence sum average)",
        "SumVariance": "Sum variance (co-occurrence sum variance)",
        "SumEntropy": "Sum entropy (co-occurrence sum entropy)",
        "Entropy": "Entropy (randomness of intensities)",
        "DifferenceVariance": "Difference variance (co-occurrence difference variance)",
        "DifferenceEntropy": "Difference entropy (co-occurrence difference entropy)",
        "InfoMeas1": "Information measure of correlation 1",
        "InfoMeas2": "Information measure of correlation 2",
        "HaralickCorrelation": "Haralick correlation",
    }
    return m.get(feat, feat)


def humanise_areas_shape(name: str) -> str:
    """Human descriptions for AreaShape fields (Zernike handled generically)."""
    m = {
        "Area": "Area of the object in pixels",
        "Perimeter": "Perimeter length of the object in pixels",
        "Eccentricity": "Eccentricity (0 = circle, 1 = line)",
        "Solidity": "Solidity (area / convex area)",
        "Extent": "Extent (object area / bounding box area)",
        "Compactness": "Compactness (perimeter² / 4π area)",
        "FormFactor": "Form factor (4π area / perimeter²)",
        "MajorAxisLength": "Length of the major axis (best-fit ellipse)",
        "MinorAxisLength": "Length of the minor axis (best-fit ellipse)",
        "Orientation": "Orientation angle of the major axis (degrees)",
        "FeretDiameter": "Maximum caliper (Feret) diameter",
        "ConvexArea": "Area of the convex hull of the object",
        "EulerNumber": "Euler number (components minus holes)",
        "BoundingBoxArea": "Area of the tight axis-aligned bounding box",
        "Zernike": "Zernike moment coefficient (order,moment) describing shape",
    }
    if name.startswith("Zernike"):
        return m["Zernike"]
    return m.get(name, name.replace("_", " "))


def build_rules() -> List[PatternRule]:
    """Define regex rules that cover most CellProfiler feature families."""
    R: List[PatternRule] = []

    R.append(PatternRule(
        name="AreaShape_generic",
        regex=re.compile(r"^AreaShape_(?P<measure>[A-Za-z]+(?:_[0-9]+_[0-9]+)?)$"),
        template="Area/shape feature: {measure_desc}.",
        family="AreaShape",
    ))

    R.append(PatternRule(
        name="Intensity_stat_channel",
        regex=re.compile(r"^Intensity_(?P<stat>[A-Za-z]+Intensity(?:Edge)?)_(?P<channel>.+)$"),
        template="{stat_desc} in channel '{channel}'.",
        family="Intensity",
    ))
    R.append(PatternRule(
        name="Intensity_total_like",
        regex=re.compile(r"^Intensity_(?P<stat>TotalIntensity|MassDisplacement)_(?P<channel>.+)$"),
        template="{stat_desc} for channel '{channel}'.",
        family="Intensity",
    ))

    R.append(PatternRule(
        name="Texture_standard",
        regex=re.compile(r"^Texture_(?P<feature>[A-Za-z0-9]+)_(?P<channel>[^_]+)_(?P<scale>[0-9]+)_(?P<angle>[0-9]+)$"),
        template="Texture: {texture_desc} on channel '{channel}' at scale {scale} and angle {angle}°.",
        family="Texture",
    ))

    R.append(PatternRule(
        name="Granularity_channel_scale",
        regex=re.compile(r"^Granularity_(?P<channel>[^_]+)_(?P<scale>[0-9]+)$"),
        template="Granularity at radius {scale} on channel '{channel}'.",
        family="Granularity",
    ))

    R.append(PatternRule(
        name="RadialDistribution_frac",
        regex=re.compile(r"^RadialDistribution_FracAtD_(?P<channel>[^_]+)_(?P<bin>[0-9]+)of(?P<tot>[0-9]+)$"),
        template="Radial distribution: fraction of intensity at annulus {bin} of {tot} in channel '{channel}'.",
        family="RadialDistribution",
    ))
    R.append(PatternRule(
        name="RadialDistribution_summary",
        regex=re.compile(r"^RadialDistribution_(?P<stat>MeanFrac|RadialCV)_(?P<channel>.+)$"),
        template="Radial distribution summary ({stat}) for channel '{channel}'.",
        family="RadialDistribution",
    ))

    R.append(PatternRule(
        name="Correlation_between_channels",
        regex=re.compile(r"^Correlation_(?P<metric>Correlation)_(?P<ch1>[^_]+)_(?P<ch2>[^_]+)$"),
        template="Channel correlation ({metric}) between '{ch1}' and '{ch2}'.",
        family="Correlation",
    ))

    R.append(PatternRule(
        name="Location_basic",
        regex=re.compile(r"^Location_(?P<field>Center_X|Center_Y|Center_Z|MaximumRadius|MinimumRadius|Eccentricity)$"),
        template="Object location/geometry field: {field}.",
        family="Location",
    ))
    R.append(PatternRule(
        name="Location_center_of_mass_channel",
        regex=re.compile(r"^Location_CenterMassIntensity_(?P<axis>X|Y|Z)_(?P<channel>.+)$"),
        template="Intensity-weighted centre-of-mass {axis}-coordinate for channel '{channel}'.",
        family="Location",
    ))

    R.append(PatternRule(
        name="Neighbors_generic",
        regex=re.compile(r"^Neighbors_(?P<metric>FirstClosestDistance|SecondClosestDistance|AngleBetweenNeighbors_[0-9]+|NumberOfNeighbors)$"),
        template="Neighbourhood metric: {metric}.",
        family="Neighbors",
    ))

    R.append(PatternRule(
        name="ImageQuality_generic",
        regex=re.compile(r"^ImageQuality_(?P<metric>[A-Za-z0-9]+)(?:_(?P<channel>.+))?$"),
        template="Image quality metric: {metric}" + " for channel '{channel}'." + " if channel present.",
        family="ImageQuality",
    ))

    R.append(PatternRule(
        name="Count_objects",
        regex=re.compile(r"^Count_(?P<object>.+)$"),
        template="Count of objects: '{object}' per image or parent.",
        family="Count",
    ))

    R.append(PatternRule(
        name="Parent_link",
        regex=re.compile(r"^Parent_(?P<parent>.+)$"),
        template="Parent object identifier for '{parent}'.",
        family="Parent",
    ))
    R.append(PatternRule(
        name="Children_count",
        regex=re.compile(r"^Children_(?P<child>.+)_Count$"),
        template="Number of child objects of type '{child}'.",
        family="Children",
    ))

    R.append(PatternRule(
        name="Threshold_generic",
        regex=re.compile(r"^Threshold_(?P<field>.+)$"),
        template="Thresholding diagnostic/value: {field}.",
        family="Threshold",
    ))

    R.append(PatternRule(
        name="AreaOccupied_generic",
        regex=re.compile(r"^AreaOccupied_(?P<field>.+)$"),
        template="Area occupied metric: {field}.",
        family="AreaOccupied",
    ))

    R.append(PatternRule(
        name="ObjectNumber",
        regex=re.compile(r"^ObjectNumber$"),
        template="Unique identifier for the object instance.",
        family="General",
    ))

    return R


def apply_rule(rule: PatternRule, feature: str) -> Optional[str]:
    """Apply a rule to a feature name and return a human-readable description or None."""
    m = rule.regex.match(feature)
    if not m:
        return None
    d = m.groupdict()

    if rule.family == "AreaShape" and "measure" in d:
        d["measure_desc"] = humanise_areas_shape(d["measure"])
    if rule.family == "Intensity" and "stat" in d:
        d["stat_desc"] = humanise_stat(d["stat"])
    if rule.family == "Texture" and "feature" in d:
        d["texture_desc"] = humanise_texture(d["feature"])

    desc = rule.template.format(**d)
    desc = desc.replace(" for channel 'None'.", "").replace(" if channel present.", "")
    return desc


def write_patterns(rules: List[PatternRule], out_path: str) -> None:
    """Write the regex pattern dictionary to TSV."""
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("pattern\tfamily\tdescription_template\n")
        for r in rules:
            fh.write(f"{r.regex.pattern}\t{r.family}\t{r.template}\n")


def read_columns(scan_file: str, delimiter: str = "\t") -> List[str]:
    """Read only the header/column names from a dataset file."""
    ext = os.path.basename(scan_file).lower()
    if ext.endswith(".parquet"):
        if pd is None:
            raise RuntimeError("pandas is required to read Parquet.")
        return list(pd.read_parquet(scan_file, columns=[]).columns)
    if ext.endswith(".feather"):
        if pd is None:
            raise RuntimeError("pandas is required to read Feather.")
        return list(pd.read_feather(scan_file).columns)
    if ext.endswith(".tsv.gz") or ext.endswith(".txt.gz") or ext.endswith(".csv.gz"):
        opener = gzip.open
        mode = "rt"
    else:
        opener = open
        mode = "r"

    with opener(scan_file, mode, encoding="utf-8", errors="ignore") as fh:
        header = fh.readline().rstrip("\n\r")
    return header.split(delimiter)



def read_feature_list(path: str, sep: str = "\t") -> List[str]:
    """Read features from a text file. If the file has one line, split by sep;
    otherwise treat each non-empty line as a feature name."""
    feats: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = [ln.rstrip("\n\r") for ln in fh.readlines()]
    lines = [ln for ln in lines if ln.strip() != ""]
    if not lines:
        return feats
    if len(lines) == 1:
        feats = lines[0].split(sep)
    else:
        feats = lines
    return feats

def expand_features(rules: List[PatternRule], columns: Iterable[str]) -> List[Tuple[str, str, str]]:
    """Expand concrete feature descriptions by matching patterns to provided column names."""
    out: List[Tuple[str, str, str]] = []
    for col in columns:
        if not col or col.startswith("Metadata_") or col in {"ImageNumber", "ObjectNumber", "ImageNumber_x", "ImageNumber_y"}:
            continue
        desc = None
        fam = ""
        for r in rules:
            d = apply_rule(r, col)
            if d is not None:
                desc = d
                fam = r.family
                break
        if desc is None:
            family = col.split("_", 1)[0]
            desc = f"{family} feature '{col}'."
            fam = family
        out.append((col, desc, fam))
    seen = set()
    uniq = []
    for f, d, fam in out:
        if f not in seen:
            seen.add(f)
            uniq.append((f, d, fam))
    return uniq


def write_features(rows: List[Tuple[str, str, str]], out_path: str) -> None:
    """Write concrete feature->description mapping as TSV."""
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("feature\tdescription\tfamily\n")
        for f, d, fam in rows:
            fh.write(f"{f}\t{d}\t{fam}\n")


# ------------------------------------------------------------------------------------------
# Scrape mode helpers (optional; works with local HTML too)
# ------------------------------------------------------------------------------------------

def _normalise_unicode(text: str) -> str:
    """Normalise common unicode punctuation and nbsp to ASCII-friendly forms."""
    if not isinstance(text, str):
        return text
    repl = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u00A0": " ",
        "\u2009": " ",
    }
    for src, tgt in repl.items():
        text = text.replace(src.encode("utf-8").decode("unicode_escape"), tgt)
    return text


def make_session() -> "requests.Session":
    """Create a polite requests session with retries."""
    if requests is None:
        raise RuntimeError("requests is required for scrape mode but is not installed.")
    s = requests.Session()
    s.headers.update({
        "User-Agent": "CP-Features-Dict/1.0 (+https://cellprofiler.org)",
        "Accept": "text/html,application/xhtml+xml",
    })
    retries = requests.adapters.Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def list_module_pages_online(base_url: str, session: "requests.Session") -> List[str]:
    """Collect /modules/*.html links by crawling from the version index (depth 2)."""
    index_url = urljoin(base_url, "index.html")
    r = session.get(index_url, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "lxml")
    candidates = set()

    def collect_modules_from(soup_obj, base):
        for a in soup_obj.select("a[href]"):
            href = a.get("href", "")
            if "/modules/" in href and href.endswith(".html"):
                candidates.add(urljoin(base, href))

    collect_modules_from(soup, index_url)

    for u in list(candidates):
        try:
            rr = session.get(u, timeout=20)
            if rr.status_code != 200:
                continue
            ss = BeautifulSoup(rr.content, "lxml")
            collect_modules_from(ss, u)
        except Exception:
            continue

    cleaned = []
    seen = set()
    for u in sorted(candidates):
        if u.endswith("/index.html"):
            continue
        if u not in seen:
            seen.add(u)
            cleaned.append(u)
    return cleaned


def list_module_pages_offline(html_dir: str) -> List[str]:
    """Return file:// URLs for module HTML files in a local directory (or root)."""
    from pathlib import Path as _Path
    paths = []
    p = _Path(html_dir)
    if p.is_dir():
        for fp in p.rglob("*.html"):
            if fp.name.lower() == "index.html":
                # keep, as some modules are index pages
                pass
            paths.append(fp.resolve().as_uri())
    return sorted(paths)


def fetch_html(url: str, session: Optional["requests.Session"]) -> str:
    """Fetch a URL or file:// URL and return HTML text, skipping XML listings."""
    parsed = urlparse(url)
    if parsed.scheme == "file":
        with open(parsed.path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    if session is None:
        raise RuntimeError("Attempted to fetch online content without a requests session.")
    r = session.get(url, timeout=30)
    r.raise_for_status()
    txt = r.text
    if txt.lstrip().startswith("<?xml") and "ListBucketResult" in txt:
        return "<html></html>"
    return txt


def extract_measurement_section(soup: "BeautifulSoup") -> List["BeautifulSoup"]:
    """Locate the 'Measurements made by this module' section (or similar)."""
    patterns = [
        re.compile(r"^Measurements\s+made\s+by\s+this\s+module", re.I),
        re.compile(r"^Measurements", re.I),
        re.compile(r"^What\s+measurements", re.I),
    ]
    heading = None
    for tag in soup.find_all(re.compile("h[1-6]")):
        txt = tag.get_text(" ", strip=True)
        if any(p.search(txt or "") for p in patterns):
            heading = tag
            break
    if not heading:
        return []
    collected = []
    level = int(heading.name[1])
    for sib in heading.next_siblings:
        if getattr(sib, "name", None) and re.fullmatch(r"h[1-6]", sib.name or ""):
            if int(sib.name[1]) <= level:
                break
        if getattr(sib, "name", None) in {"p", "ul", "ol", "table", "div"}:
            collected.append(sib)
    return collected


def parse_features_from_elements(elements: Iterable["BeautifulSoup"]) -> List[Tuple[str, str]]:
    """Parse features from tables, lists, and paragraphs within the measurement section."""
    out: List[Tuple[str, str]] = []

    def normalise_feature(text: str) -> str:
        text = _normalise_unicode(html.unescape(text or "").strip())
        text = re.sub(r"\s+", " ", text)
        text = text.rstrip(" :;,.")
        return text

    header_like = {"feature", "features", "measurement", "measurement name", "name", "measure", "measurements"}

    for el in elements:
        if el.name == "table":
            rows = el.find_all("tr")
            for tr in rows:
                cols = tr.find_all(["td", "th"])
                if len(cols) < 2:
                    continue
                left = normalise_feature(cols[0].get_text(" ", strip=True)).lower()
                right_raw = _normalise_unicode(cols[1].get_text(" ", strip=True))
                if left in header_like or not left:
                    continue
                feature = normalise_feature(cols[0].get_text(" ", strip=True))
                desc = right_raw.strip()
                if feature and desc:
                    out.append((feature, desc))

        if el.name in {"ul", "ol"}:
            for li in el.find_all("li"):
                txt = _normalise_unicode(li.get_text(" ", strip=True))
                if ":" in txt:
                    feat, desc = txt.split(":", 1)
                    feat = normalise_feature(feat)
                    desc = desc.strip()
                    if len(feat) >= 3 and len(desc) >= 3:
                        out.append((feat, desc))

        if el.name == "p":
            strong = el.find(["b", "strong"])
            if strong and ":" in el.get_text(" ", strip=True):
                txt = _normalise_unicode(el.get_text(" ", strip=True))
                feat, desc = txt.split(":", 1)
                feat = normalise_feature(feat)
                desc = desc.strip()
                if len(feat) >= 3 and len(desc) >= 3:
                    out.append((feat, desc))

    seen = set()
    unique = []
    for feat, desc in out:
        key = (feat.lower(), desc)
        if key not in seen:
            seen.add(key)
            unique.append((feat, desc))
    return unique


def scrape_module(url: str, session: Optional["requests.Session"]) -> List[FeatureRow]:
    """Scrape a single module page for features."""
    if BeautifulSoup is None:
        return []
    html_txt = fetch_html(url, session)
    soup = BeautifulSoup(html_txt, "lxml")
    title = soup.find("h1")
    module = (title.get_text(" ", strip=True) if title else "") or os.path.splitext(os.path.basename(urlparse(url).path))[0]
    module = re.sub(r"\s+", " ", module)
    elements = extract_measurement_section(soup)
    pairs = parse_features_from_elements(elements)
    return [FeatureRow(feature=f, description=d, module=module, source_url=url) for f, d in pairs]


def generic_prefix_rows() -> List[FeatureRow]:
    """Curated generic prefix descriptions to act as fallbacks in scrape mode."""
    items = [
        ("AreaShape_", "Object morphology metrics (area, perimeter, eccentricity, solidity, etc.)."),
        ("Location_", "Object centroid and related spatial coordinates."),
        ("Intensity_", "Per-object intensity statistics per channel (mean, median, min, max, integrated, etc.)."),
        ("Granularity_", "Multi-scale granularity features describing texture at different radii."),
        ("Texture_", "Haralick/Grey-level co-occurrence texture features per scale and angle."),
        ("RadialDistribution_", "Radial distribution of intensity from object centre to edge."),
        ("Neighbors_", "Neighbourhood and adjacency counts/distances between objects."),
        ("Correlation_", "Correlation of intensities between image channels."),
        ("ImageQuality_", "Per-image quality control metrics (focus, saturation, blur, noise)."),
        ("Count_", "Object counts per image or per parent object."),
        ("Parent_", "Links to parent object identifiers."),
        ("Children_", "Links to child object counts/identifiers."),
        ("ModuleError_", "Module error flags or diagnostics."),
        ("Threshold_", "Per-image/module threshold values and diagnostics."),
        ("ObjectNumber", "Unique identifier for each object instance."),
        ("AreaOccupied_", "Area occupied by objects in the image (mask coverage)."),
        ("Speed_", "Object motion or displacement metrics across frames (if time-series)."),
    ]
    return [FeatureRow(feature=f, description=d, module="(generic)", source_url="") for f, d in items]


# ------------------------------------------------------------------------------------------
# CLI + main
# ------------------------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Create and return the command-line argument parser."""
    p = argparse.ArgumentParser(
        description="Unified tool: scrape CellProfiler manual OR expand from your dataset header to build a feature dictionary (TSV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", type=str, default="", choices=["scrape", "expand"], help="Which action to perform.")
    # Scrape mode args
    p.add_argument("--manual_version", type=str, default=DEFAULT_MANUAL_VERSION, help="Manual version for online scraping.")
    p.add_argument("--html_dir", type=str, default="", help="Local directory containing manual HTML (root or modules/).")
    p.add_argument("--include_generic_prefixes", action="store_true", help="Include curated generic feature prefixes in scrape mode.")
    p.add_argument("--out_tsv", type=str, default="", help="Output TSV for scrape mode.")
    # Expand mode args
    p.add_argument("--scan_file", type=str, default="", help="Dataset file to read header/columns from (TSV/CSV/Parquet/Feather).")
        p.add_argument("--feature_list", type=str, default="", help="Text file containing either a single header line OR newline-separated feature names.")
        p.add_argument("--feature_list_sep", type=str, default="\t", help="Delimiter to split a single-line header in --feature_list (ignored if file has multiple lines).")
    p.add_argument("--delimiter", type=str, default="\t", help="Delimiter for TSV/CSV when scanning header.")
    p.add_argument("--out_features", type=str, default="", help="Output TSV for expand mode.")
    # Always available
    p.add_argument("--out_patterns", type=str, default="", help="Optional TSV of regex patterns -> templates.")
    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    return p


def setup_logging(level: str) -> None:
    """Configure logging with the requested verbosity."""
    logging.basicConfig(level=getattr(logging, level), format="%(levelname)s: %(message)s")


def run_scrape(manual_version: str, html_dir: str, include_generic: bool, out_tsv: str) -> None:
    """Run scrape mode and write the resulting TSV."""
    rows: List[FeatureRow] = []
    if include_generic:
        rows.extend(generic_prefix_rows())

    if html_dir:
        urls = list_module_pages_offline(html_dir)
        session = None
    else:
        if requests is None or BeautifulSoup is None:
            raise RuntimeError("Online scrape requires 'requests' and 'beautifulsoup4' packages.")
        session = make_session()
        base_url = DEFAULT_BASE.format(ver=manual_version)
        urls = list_module_pages_online(base_url, session)

    total = 0
    for url in urls:
        try:
            feat_rows = scrape_module(url, session)
            rows.extend(feat_rows)
            total += len(feat_rows)
        except Exception:
            continue

    # Deduplicate
    unique = []
    seen = set()
    for r in rows:
        key = (r.feature.lower(), r.description, r.module)
        if key not in seen:
            seen.add(key)
            unique.append(r)

    if not out_tsv:
        raise RuntimeError("Please provide --out_tsv for scrape mode.")
    with open(out_tsv, "w", encoding="utf-8") as fh:
        fh.write("feature\tdescription\tmodule\tsource_url\n")
        for r in unique:
            feat = _normalise_unicode(r.feature.replace("\n", " ").strip())
            desc = _normalise_unicode(r.description.replace("\n", " ").strip())
            mod = _normalise_unicode(r.module.replace("\n", " ").strip())
            src = r.source_url.strip()
            fh.write(f"{feat}\t{desc}\t{mod}\t{src}\n")


def run_expand(scan_file: str, delimiter: str, out_features: str, feature_list: str = "", feature_list_sep: str = "\t") -> None:
    """Run expand mode and write the concrete features TSV."""
    rules = build_rules()
    if not out_features:
        raise RuntimeError("Please provide --out_features for expand mode.")
    if not scan_file:
        raise RuntimeError("Please provide --scan_file for expand mode.")
    cols = read_columns(scan_file, delimiter=delimiter)
    rows = expand_features(rules, cols)
    write_features(rows, out_features)


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    setup_logging(args.log_level)

    if args.out_patterns:
        write_patterns(build_rules(), args.out_patterns)

    if args.mode == "scrape":
        run_scrape(args.manual_version, args.html_dir, args.include_generic_prefixes, args.out_tsv)
        return

    if args.mode == "expand":
        run_expand(args.scan_file, args.delimiter, args.out_features)
        return

    # Auto-detect mode when possible for convenience
    if args.html_dir or args.out_tsv:
        run_scrape(args.manual_version, args.html_dir, args.include_generic_prefixes, args.out_tsv)
        return

    if args.scan_file or args.out_features:
        run_expand(args.scan_file, args.delimiter, args.out_features)
        return

    # If nothing provided, print help-like message
    raise SystemExit("Nothing to do. Provide --mode scrape or --mode expand (see -h).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted by user")
