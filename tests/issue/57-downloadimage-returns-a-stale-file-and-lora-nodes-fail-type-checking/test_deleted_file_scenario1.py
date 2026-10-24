"""
### 이슈
1. 폴더의 파일을 지운 뒤 새 URL을 받으면 옛 이미지가 나온다.
   * ASIS: `1.png`, `2.png`, `3.png`에서 `2.png`를 지우고 새 URL을 받으면 `3.png`(옛 이미지)를 돌려준다.
   * TOBE: 새 URL의 이미지를 돌려준다.

### 원인 분석
1. 번호를 `1 + 폴더의 파일 수`로 정해 남은 파일과 겹치고, 파일이 있으면 다운로드를 건너뛴다.

### 솔루션
1. 파일 이름을 URL 해시로 정해, 이름이 같으면 같은 URL이다.

### 테스트
1. 시나리오: URL 3개를 받고, 그중 한 파일을 지운 뒤 새 URL을 받는다.
2. 예제 1건: 지우는 파일만 바꾼다.
    첫 번째 파일 삭제  <- 기준
    두 번째 파일 삭제
    세 번째 파일 삭제
3. 비교: 새 URL로 받은 이미지가 그 URL의 이미지이고, 남은 파일은 그대로다.
"""

import pytest


@pytest.mark.parametrize("deleted", [0, 1, 2])
def test_a_new_url_after_a_deletion_gets_its_own_image(download, deleted):
    urls = [f"http://host/{i}.png" for i in range(3)]
    paths = [download(u)[1] for u in urls]
    paths[deleted].unlink()
    image, path = download("http://host/new.png")
    assert download.pixel(image) == download.color_of("http://host/new.png")
    assert all(p.is_file() for i, p in enumerate(paths) if i != deleted)
    assert path not in paths
