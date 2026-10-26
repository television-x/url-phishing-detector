import re
import math
from collections import Counter
from urllib.parse import urlparse

SUSPICIOUS_TLDS = {
    'xyz', 'top', 'tk', 'ml', 'ga', 'cf', 'gq', 'buzz', 'work',
    'monster', 'surf', 'club', 'online', 'site', 'click', 'fit', 'date'
}
TRUSTED_TLDS = {
    'com', 'org', 'net', 'edu', 'gov', 'io', 'in', 'co', 'uk', 'de', 'ca',
    'fr', 'au', 'jp', 'it', 'nl', 'info', 'biz', 'me', 'app', 'dev', 'ai',
    'tech', 'eu', 'es', 'ru', 'br', 'cn'
}
# Short keywords are matched as whole tokens (so "freecodecamp" is not "free");
# long keywords are matched as substrings (so "securebanking" is still caught).
KEYWORDS = [
    'login', 'verify', 'update', 'secure', 'account', 'banking', 'confirm',
    'password', 'signin', 'ebayisapi', 'paypal', 'free', 'lucky', 'bonus',
    'support', 'wallet', 'billing', 'authenticate', 'recovery', 'security'
]
SHORT_KW = {k for k in KEYWORDS if len(k) < 7}
LONG_KW = [k for k in KEYWORDS if len(k) >= 7]

# Registered domains (eTLD+1). Whitelist is matched on the registered domain,
# NOT with endswith(), so "google.com.evil.tk" can never match.
KNOWN_TOP_DOMAINS = {
    'google.com', 'google.co.in', 'youtube.com', 'facebook.com', 'amazon.com',
    'amazon.in', 'apple.com', 'wikipedia.org', 'microsoft.com', 'github.com',
    'netflix.com', 'linkedin.com', 'instagram.com', 'reddit.com', 'twitter.com',
    'x.com', 'yahoo.com', 'stackoverflow.com', 'whatsapp.com', 'zoom.us',
    'bing.com', 'twitch.tv', 'adobe.com', 'dropbox.com', 'spotify.com',
    'paypal.com', 'ebay.com', 'craigslist.org', 'live.com', 'office.com',
    'imdb.com', 'pinterest.com'
}
# Platforms where ANYONE gets a subdomain: apex is trusted, subdomains are not.
USER_CONTENT_DOMAINS = {'wordpress.com', 'tumblr.com', 'medium.com',
                        'github.io', 'blogspot.com', 'weebly.com'}

# Minimal public-suffix handling without a dependency (use `tldextract` for production).
MULTI_SUFFIX = {'co.in', 'co.uk', 'com.au', 'co.jp', 'com.br', 'com.cn',
                'org.uk', 'ac.in', 'gov.in', 'net.in', 'org.in'}

_SCHEME_RE = re.compile(r'^[a-zA-Z][a-zA-Z0-9+.\-]*://')
_IP_RE = re.compile(r'^\d{1,3}(\.\d{1,3}){3}$')
_TOKEN_RE = re.compile(r'[^a-z0-9]+')


def shannon_entropy(s: str) -> float:
    """O(n) entropy using Counter (old version was O(n*k) via s.count in a loop)."""
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def normalize_url(raw_url: str) -> str:
    """Case-insensitive scheme detection; only prepends https:// when no scheme exists."""
    url = str(raw_url).strip()
    if not _SCHEME_RE.match(url):
        url = 'https://' + url
    return url


def split_host(clean_host: str):
    """Return (subdomain_count, registered_domain, sld_label, tld)."""
    parts = clean_host.split('.') if clean_host else []
    if len(parts) < 2:
        return 0, clean_host, clean_host, ''
    suffix_len = 2 if '.'.join(parts[-2:]) in MULTI_SUFFIX else 1
    if len(parts) <= suffix_len:
        return 0, clean_host, clean_host, parts[-1]
    registered = '.'.join(parts[-(suffix_len + 1):])
    sld = parts[-(suffix_len + 1)]
    return len(parts) - (suffix_len + 1), registered, sld, parts[-1]


def registered_domain(raw_url: str) -> str:
    """Used by model.py to group train/test splits by domain (prevents leakage)."""
    host = (urlparse(normalize_url(raw_url)).hostname or '').lower()
    host = host[4:] if host.startswith('www.') else host
    return split_host(host)[1]


def _has_keyword(text: str) -> int:
    text = text.lower()
    tokens = set(_TOKEN_RE.split(text))
    return int(bool(tokens & SHORT_KW) or any(k in text for k in LONG_KW))


def _safe_port(parsed):
    try:
        return parsed.port
    except ValueError:      # malformed port, e.g. http://a.com:abc
        return -1


def extract_features(raw_url: str):
    url = normalize_url(raw_url)
    parsed = urlparse(url)
    hostname = (parsed.hostname or '').lower()
    path = parsed.path or ''
    clean_host = hostname[4:] if hostname.startswith('www.') else hostname

    sub_count, reg_domain, sld, tld = split_host(clean_host)
    is_ip = bool(_IP_RE.match(clean_host))
    if is_ip:
        sub_count = 0   # an IP has no subdomains

    is_top = int(
        reg_domain in KNOWN_TOP_DOMAINS
        and not (reg_domain in USER_CONTENT_DOMAINS and sub_count > 0)
    )
    port = _safe_port(parsed)

    return [
        len(url),                                                  # 1
        len(clean_host),                                           # 2
        len(path),                                                 # 3
        clean_host.count('.'),                                     # 4
        clean_host.count('-'),                                     # 5
        int('@' in url),                                           # 6
        int(parsed.scheme == 'https'),                             # 7
        sub_count,                                                 # 8
        int(is_ip),                                                # 9
        sum(c.isdigit() for c in clean_host),                      # 10
        sum(c.isdigit() for c in url) / max(len(url), 1),          # 11
        int(not is_top and _has_keyword(clean_host)),              # 12
        int(not is_top and _has_keyword(path)),                    # 13
        int(tld in SUSPICIOUS_TLDS),                               # 14
        int(tld in TRUSTED_TLDS),                                  # 15
        sum(url.count(c) for c in '?=&%'),                         # 16
        shannon_entropy(sld),                                      # 17 entropy of the label, not the whole host
        path.count('/'),                                           # 18
        int(port not in (None, 80, 443)),                          # 19 (malformed port counts as suspicious)
        is_top,                                                    # 20
    ]


FEATURE_NAMES = [
    "URL Length", "Hostname Length", "Path Length", "Dots in Hostname",
    "Hyphens in Hostname", "Has @ Symbol", "Has HTTPS", "Subdomain Count",
    "Has IP Address", "Digits in Hostname", "Digit Ratio in URL",
    "Keyword in Hostname", "Keyword in Path", "Suspicious TLD", "Trusted TLD",
    "Special Char Count", "Hostname Entropy", "Slash Count", "Has Custom Port",
    "Is Top Domain",
]
assert len(FEATURE_NAMES) == 20