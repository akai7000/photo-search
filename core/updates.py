"""Discover published updates from the official downloads page."""

import re
import ssl
from html.parser import HTMLParser
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import certifi

from core.data_location import load_config

DOWNLOADS_PAGE = "https://www.snowfoxninja.com/"
DEFAULT_DOWNLOAD_URL = "https://www.snowfoxninja.com/#downloads"
INSTALLER_HOST = "pub-5c62851ff87d40acaeee8eb31d082ac7.r2.dev"
INSTALLER_PATH = re.compile(
    r"/installers/PhotoSearch-([0-9]+(?:\.[0-9]+){1,3})-CPU-Setup\.exe",
    re.IGNORECASE,
)
VERSION = re.compile(r"[0-9]+(?:\.[0-9]+){1,3}")
MAX_PAGE_BYTES = 1024 * 1024
REQUEST_TIMEOUT = 5


def _version_key(version: str) -> tuple[int, ...]:
    if not VERSION.fullmatch(version):
        raise ValueError(f"Invalid release version: {version}")
    parts = [int(part) for part in version.split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


class _InstallerLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.installers = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if not href:
            return
        url = urlsplit(href)
        if url.scheme != "https" or url.netloc.lower() != INSTALLER_HOST:
            return
        match = INSTALLER_PATH.fullmatch(url.path)
        if match:
            version = match.group(1)
            self.installers.append((_version_key(version), version, href))


def load_download_url() -> str:
    url = load_config().get("download_url", DEFAULT_DOWNLOAD_URL)
    if not isinstance(url, str) or any(char.isspace() or ord(char) < 32 for char in url):
        raise ValueError("Invalid download_url in local configuration")
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("download_url must be an absolute HTTPS URL without credentials")
    _ = parsed.port
    return url


def check_for_update(current_version: str) -> tuple[str, str] | None:
    current = _version_key(current_version)
    headers = {"User-Agent": f"PhotoSearch/{current_version}"}
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(Request(DOWNLOADS_PAGE, headers=headers), timeout=REQUEST_TIMEOUT, context=context) as response:
        content = response.read(MAX_PAGE_BYTES + 1)
    if len(content) > MAX_PAGE_BYTES:
        raise ValueError("Downloads page exceeds the update check size limit")
    parser = _InstallerLinks()
    parser.feed(content.decode("utf-8"))
    if not parser.installers:
        raise ValueError("No CPU R2 installer links found on the downloads page")
    for version_key, version, url in sorted(parser.installers, key=lambda item: item[0], reverse=True):
        if version_key <= current:
            break
        try:
            with urlopen(Request(url, headers=headers, method="HEAD"), timeout=REQUEST_TIMEOUT, context=context):
                return version, load_download_url()
        except HTTPError as exc:
            if exc.code != 404:
                raise
    return None
