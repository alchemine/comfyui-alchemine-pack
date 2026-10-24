"""
### 이슈
1. 큰 데이터셋을 나눠 읽을 수 없고, 점이 있는 이름은 잘려서 저장된다.

### 원인 분석
1. `LoadImagesFromDir`에 읽을 범위 입력이 없고, `SaveImageWithText`가 `with_suffix`로 확장자를 바꾼다.

### 솔루션
1. `count`, `chunk`로 일부만 읽고, 확장자는 stem 뒤에 붙인다.

### 테스트
1. 시나리오: 다운로더가 타임스탬프 이름으로 받은 이미지를 나눠 읽어 캡션과 저장한다.
2. 예제 5건(모두 직접 추가):
    1.000Z_1.jpg 를 저장
    2024-10-24T09.30.00.000Z_2.png 를 저장
    sdxl_v1.0.png 를 저장
    폴더 5장을 count=2, chunk=1로 읽기
    폴더 5장을 count=3, chunk=1로 읽기
3. 비교: 저장 이름이 원본 stem을 유지하고, 읽은 묶음이 기대한 파일만 담는다.
"""

import pytest


@pytest.mark.parametrize(
    "name", ["1.000Z_1.jpg", "2024-10-24T09.30.00.000Z_2.png", "sdxl_v1.0.png"]
)
def test_timestamp_names_keep_their_stem(save, name):
    stem = name.rsplit(".", 1)[0]
    assert save(name=name)[0] == f"{save.dir.name}/{stem}.png"


@pytest.mark.parametrize(
    "count, expected", [(2, ["03.png", "04.png"]), (3, ["04.png", "05.png"])]
)
def test_the_second_chunk_holds_only_its_images(pack, dataset, count, expected):
    assert (
        pack("lora").LoadImagesFromDir.execute(dataset(5), count=count, chunk=1)[1]
        == expected
    )
