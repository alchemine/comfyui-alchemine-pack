"""
### 이슈
1. 큰 데이터셋 폴더를 나눠 읽을 수 없다.
   * ASIS: `LoadImagesFromDir`는 폴더의 이미지를 항상 전부 읽는다.
   * TOBE: `count`장씩, `chunk`번째 묶음만 읽는다.

### 원인 분석
1. 노드에 읽을 범위를 정하는 입력이 없다.

### 솔루션
1. `count`, `chunk` 선택 입력으로 정렬된 이미지 목록의 일부만 읽는다.

### 테스트
1. 시나리오: 이미지 5장 폴더를 `count=2`로 두고 `chunk`를 0부터 올리며 워크플로를 실행한다.
2. 예제 1건: 같은 폴더에서 `chunk`만 바꾼다.
    chunk=0  <- 기준
    chunk=1
    chunk=2
3. 비교: 세 번의 실행이 모든 이미지를 한 번씩, 순서대로 읽고, 실행마다 `IS_CHANGED` 값이 다르다.
"""


def test_queueing_each_chunk_reads_every_image_once(pack, dataset):
    node = pack("lora").LoadImagesFromDir
    folder = dataset(5)
    names, keys = [], []
    for chunk in range(3):
        images, chunk_names = node.execute(folder, count=2, chunk=chunk)
        assert len(images) == len(chunk_names)
        names += chunk_names
        keys.append(node.IS_CHANGED(folder, count=2, chunk=chunk))
    assert names == ["01.png", "02.png", "03.png", "04.png", "05.png"]
    assert len(set(keys)) == 3
