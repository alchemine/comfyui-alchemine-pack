"""
### 이슈
1. 같은 URL을 다시 받으면 파일이 또 생긴다.
   * ASIS: 서버를 다시 켜서 캐시가 비면 같은 URL을 `2.png`로 다시 내려받는다.
   * TOBE: 이미 받은 파일을 그대로 쓴다.

### 원인 분석
1. 파일 이름이 URL과 상관없는 번호라 이미 받았는지 알 수 없다.

### 솔루션
1. 파일 이름을 URL 해시로 정해, 같은 URL이면 있는 파일을 읽는다.

### 테스트
1. 시나리오: 같은 URL로 노드를 여러 번 실행한다(캐시 없이 `execute`를 직접 부른다).
2. 예제 1건: URL마다 1건이다(확장자, 쿼리 유무가 다르다).
3. 비교: 실행마다 같은 파일을 돌려주고, 다운로드는 한 번만 하며, 폴더에 파일이 하나다.
"""

import pytest

URLS = [
    "http://host/a.png",
    "http://host/a.jpg?w=512",
    "http://host/image",
    "https://cdn.donmai.us/original/ab/cd/abcd.jpg",
]


@pytest.mark.parametrize("url", URLS)
def test_running_twice_reuses_the_file(download, url):
    first = download(url)[1]
    second = download(url)[1]
    assert first == second
    assert download.fetched == [url]
    assert list(download.dir.iterdir()) == [first]
