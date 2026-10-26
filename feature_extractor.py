import re
import math
from urllib.parse import urlparse

# High-abuse TLDs frequently used in phishing attacks
SUSPICIOUS_TLDS = {
    '.xyz', '.top', '.tk', '.ml', '.ga', '.cf', '.gq', '.buzz', '.work',
    '.monster', '.surf', '.club', '.online', '.site', '.click', '.fit', '.date'
}

# Standard, well-established TLDs and country codes
TRUSTED_TLDS = [
    '.com', '.org', '.net', '.edu', '.gov', '.io', '.in', '.co', '.uk',
    '.de', '.ca', '.fr', '.au', '.jp', '.it', '.nl', '.info', '.biz', '.me',
    '.app', '.dev', '.ai', '.tech', '.eu', '.es', '.ru', '.br', '.cn'
]

# Sensitive brand/action keywords commonly used in credential harvesting
SUSPICIOUS_KEYWORDS = [
    'login', 'verify', 'update', 'secure', 'account',
    'banking', 'confirm', 'password', 'signin', 'ebayisapi',
    'paypal', 'free', 'lucky', 'bonus', 'support', 'wallet',
    'billing', 'authenticate', 'recovery', 'security'
]

# Curated list of verified major global web domains
KNOWN_TOP_DOMAINS = {
    'google.com', 'google.co.in', 'youtube.com', 'facebook.com', 'amazon.com',
    'amazon.in', 'apple.com', 'wikipedia.org', 'microsoft.com', 'github.com',
    'netflix.com', 'linkedin.com', 'instagram.com', 'reddit.com', 'twitter.com',
    'x.com', 'yahoo.com', 'stackoverflow.com', 'whatsapp.com', 'zoom.us',
    'bing.com', 'twitch.tv', 'adobe.com', 'dropbox.com', 'spotify.com',
    'paypal.com', 'ebay.com', 'craigslist.org', 'live.com', 'office.com',
    'medium.com', 'imdb.com', 'pinterest.com', 'tumblr.com', 'wordpress.com'
}


def shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy to detect DGA / randomized domain names."""
    if not s:
        return 0.0
    prob = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob)


def normalize_url(raw_url: str) -> str:
    """
    Ensure the URL has a scheme for reliable parsing.
    Handles schemeless inputs like 'google.com' or 'www.google.com'.
    """
    url = str(raw_url).strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url


def extract_features(raw_url: str):
    """
    Extract 20 lexical, structural, and domain-reputation features from a URL.
    """
    url = normalize_url(raw_url)
    parsed = urlparse(url)
    hostname = (parsed.hostname or '').lower()
    path = parsed.path or ''

    # Clean hostname: remove leading 'www.' so 'www.google.com' and 'google.com' match
    clean_host = hostname[4:] if hostname.startswith('www.') else hostname
    parts = clean_host.split('.') if clean_host else []

    # Check if registered domain belongs to known verified top entities
    is_top_domain = 0
    for td in KNOWN_TOP_DOMAINS:
        if clean_host == td or clean_host.endswith('.' + td):
            is_top_domain = 1
            break

    features = []

    # 1. URL Length — Phishing URLs often contain long tracking / spoof tokens
    features.append(len(url))

    # 2. Hostname Length — Long hostnames with multiple levels are suspicious
    features.append(len(clean_host))

    # 3. Path Length — Long redirect paths are common in phishing
    features.append(len(path))

    # 4. Dot count in hostname
    features.append(clean_host.count('.'))

    # 5. Hyphen count in hostname — Mimic domains often use hyphens (e.g. pay-pal-login)
    features.append(clean_host.count('-'))

    # 6. '@' symbol in URL — Browser ignores content prior to @, used to deceive
    features.append(1 if '@' in url else 0)

    # 7. Has HTTPS protocol
    features.append(1 if parsed.scheme == 'https' else 0)

    # 8. Subdomain count (excluding www) — e.g. google.com = 0, mail.google.com = 1
    features.append(max(len(parts) - 2, 0))

    # 9. Has IP address instead of domain name
    ip_pattern = re.compile(r'^\d{1,3}(\.\d{1,3}){3}$')
    features.append(1 if ip_pattern.match(clean_host) else 0)

    # 10. Digit count in hostname — Mimic names like paypa1.com
    features.append(sum(c.isdigit() for c in clean_host))

    # 11. Ratio of digits in overall URL
    features.append(sum(c.isdigit() for c in url) / max(len(url), 1))

    # 12. Sensitive keyword in hostname (only suspicious if NOT a verified top domain)
    has_kw_host = 1 if (is_top_domain == 0 and any(kw in clean_host for kw in SUSPICIOUS_KEYWORDS)) else 0
    features.append(has_kw_host)

    # 13. Sensitive keyword in path (only suspicious if NOT a verified top domain)
    has_kw_path = 1 if (is_top_domain == 0 and any(kw in path.lower() for kw in SUSPICIOUS_KEYWORDS)) else 0
    features.append(has_kw_path)

    # 14. Uses known suspicious / high-abuse TLD
    features.append(1 if any(clean_host.endswith(tld) for tld in SUSPICIOUS_TLDS) else 0)

    # 15. Uses standard trusted TLD
    features.append(1 if any(clean_host.endswith(tld) for tld in TRUSTED_TLDS) else 0)

    # 16. Special query character count (?, =, &, %)
    features.append(url.count('?') + url.count('=') + url.count('&') + url.count('%'))

    # 17. Shannon entropy of hostname (detects random algorithmic generation)
    features.append(shannon_entropy(clean_host))

    # 18. Slashes in path
    features.append(path.count('/'))

    # 19. Non-standard port specified
    features.append(1 if (parsed.port and parsed.port not in (80, 443)) else 0)

    # 20. Known top domain entity
    features.append(is_top_domain)

    return features


FEATURE_NAMES = [
    "URL Length",
    "Hostname Length",
    "Path Length",
    "Dots in Hostname",
    "Hyphens in Hostname",
    "Has @ Symbol",
    "Has HTTPS",
    "Subdomain Count",
    "Has IP Address",
    "Digits in Hostname",
    "Digit Ratio in URL",
    "Keyword in Hostname",
    "Keyword in Path",
    "Suspicious TLD",
    "Trusted TLD",
    "Special Char Count",
    "Hostname Entropy",
    "Slash Count",
    "Has Custom Port",
    "Is Top Domain"
]