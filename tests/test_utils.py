from lib.utils import simplify_embedded_urls


def test_simplify_embedded_urls() -> None:
    TEST_PAIRS = [
        (
            (
                "[Link Text](https://google.com/Thing?Stuff) and then another url"
                " i.reddit.com/blarg and some trailing text as well"
            ),
            (
                "[Link Text](googlecom) and then another url"
                " iredditcom and some trailing text as well"
            ),
        ),
        ("www.google.com/thingy Leading url", "wwwgooglecom Leading url"),
        ("http://only.url.com/foo?bar", "onlyurlcom"),
        ("", ""),
    ]
    for (initial, ideal) in TEST_PAIRS:
        assert simplify_embedded_urls(initial) == ideal
