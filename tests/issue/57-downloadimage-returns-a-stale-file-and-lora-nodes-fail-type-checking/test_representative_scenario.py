"""
### 이슈
1. `DownloadImage`가 옛 파일을 돌려주거나, 같은 URL을 다시 받거나, 확장자를 잘못 정한다.

### 원인 분석
1. 파일 이름을 폴더의 파일 수로 매기고, 확장자를 URL 전체의 마지막 점 뒤로 정한다.

### 솔루션
1. 파일 이름은 URL 해시, 확장자는 URL 경로의 확장자로 정한다.

### 테스트
1. 시나리오: 사용자 지적 사례를 그대로 재현한다.
2. 예제 5건:
    1.png, 2.png, 3.png 중 2.png 삭제 후 새 URL 받기  <- 사용자 지적
    같은 URL 두 번 받기                                 <- 사용자 지적
    a.jpg?w=512 받기                                    <- 사용자 지적
    a.png?v=1.2 받기                                    <- 직접 추가
    a.webp#top 받기                                     <- 직접 추가
3. 비교: 새 URL의 이미지를 돌려주고, 같은 URL은 한 번만 받고, 확장자가 URL 경로의 것이다.
"""

import pytest


def test_a_deleted_middle_file_does_not_return_a_stale_image(download):
    paths = [download(f"http://host/{i}.png")[1] for i in (1, 2, 3)]
    paths[1].unlink()
    image, _ = download("http://host/4.png")
    assert download.pixel(image) == download.color_of("http://host/4.png")


def test_the_same_url_is_fetched_once(download):
    download("http://host/a.png")
    download("http://host/a.png")
    assert download.fetched == ["http://host/a.png"]


@pytest.mark.parametrize(
    "url, suffix",
    [
        ("http://host/a.jpg?w=512", ".jpg"),
        ("http://host/a.png?v=1.2", ".png"),
        ("http://host/a.webp#top", ".webp"),
    ],
)
def test_the_query_and_fragment_stay_out_of_the_suffix(download, url, suffix):
    assert download(url)[1].suffix == suffix
