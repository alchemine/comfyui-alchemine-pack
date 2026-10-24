"""
### 이슈
1. 점이 있는 이름으로 저장하면 이름이 잘린다.
   * ASIS: `name="2024-01-01T00.00.00.000Z_1.jpg"` → `2024-01-01T00.00.00.png`
   * TOBE: `2024-01-01T00.00.00.000Z_1.png`

### 원인 분석
1. `Path.with_suffix`는 마지막 점 뒤를 확장자로 보고 바꾼다. stem에 점이 남아 있으면 그 뒤가 잘린다.

### 솔루션
1. 확장자를 바꾸지 않고 stem 뒤에 `.png`, `.txt`를 붙인다.

### 테스트
1. 시나리오: `LoadImagesFromDir.names`를 `SaveImageWithText.name`에 넣어 데이터셋을 저장한다.
2. 예제 1건: 이름마다 1건이다(점의 개수와 위치가 다르다).
3. 비교: 저장된 이미지와 캡션의 이름이 원본 stem 뒤에 확장자를 붙인 것과 같다.
"""

import pytest

NAMES = [
    "plain.jpg",
    "a.b.jpg",
    "2024-01-01T00.00.00.000Z_1.jpg",
    "v1.5_final.webp",
    "x.y.z.w.png",
    "trailing_dot..jpg",
    "1.0.png",
    "img.tar.gz.jpeg",
    "name with space.v2.png",
    "only.txt.jpg",
]


@pytest.mark.parametrize("name", NAMES)
def test_the_whole_stem_is_kept(save, name):
    stem = name.rsplit(".", 1)[0]
    image, text = save(name=name)
    assert (image, text) == (
        f"{save.dir.name}/{stem}.png",
        f"{save.dir.name}/{stem}.txt",
    )
    assert (save.dir / f"{stem}.txt").read_text() == "a, b"
