from typing import Dict
from urllib.parse import urlparse

from urlextract import URLExtract

SIMPLIFIED_CACHE: Dict[str, str] = {}


def simplify_embedded_urls(text: str) -> str:
    if text in SIMPLIFIED_CACHE:
        return SIMPLIFIED_CACHE[text]

    extractor = URLExtract()

    new_text = ""
    prev_idx = 0
    end_idx = 0
    for (url, (start_idx, end_idx)) in extractor.gen_urls(text, get_indices=True):
        # Read the snippet between the end of the last url and the start of the next.
        # Then extract the url and simplify+normalize it
        leading_snippet = text[prev_idx:start_idx]
        prev_idx = end_idx

        new_text += leading_snippet
        new_text += _simplify_url(url)

    # Add on the final snippet
    new_text += text[end_idx:]

    # Add the value to the cache
    SIMPLIFIED_CACHE[text] = new_text

    return new_text


def _simplify_url(url: str) -> str:
    # Throw on a dummy scheme if one is missing
    if "://" not in url:
        url = f"http://{url}"

    netloc = urlparse(url).netloc
    # FIXME: remove this later. This is just a hack right now so that it works with the
    # current tokenizer
    return netloc.replace(".", "")
