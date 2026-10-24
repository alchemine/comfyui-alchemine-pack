import pytest

# (url, expected suffix of the saved file)
CASES = [
    ("http://host/a.png", ".png"),
    ("http://host/a.jpg?w=512", ".jpg"),
    ("http://host/a.webp#top", ".webp"),
    ("http://host/a.JPEG", ".JPEG"),
    ("http://host/dir.v2/a.b.png", ".png"),
    ("http://host/a.png?v=1.2", ".png"),
    ("http://host/image", ""),
    ("http://host.com/a.gif?x=y&z=w.jpg", ".gif"),
    ("https://cdn.donmai.us/original/ab/cd/abcd.jpg", ".jpg"),
    ("http://host/a.png/", ".png"),
]


@pytest.mark.parametrize("url, suffix", CASES)
def test_the_suffix_comes_from_the_url_path_only(download, url, suffix):
    _, path = download(url)
    assert path.suffix == suffix and path.is_file()
