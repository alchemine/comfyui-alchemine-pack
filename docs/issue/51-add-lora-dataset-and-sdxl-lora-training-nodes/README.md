# #51 Add LoRA dataset and SDXL LoRA training nodes

이슈: https://github.com/alchemine/comfyui-alchemine-pack/issues/51

### 이슈
1. LoRA 데이터셋 준비부터 SDXL LoRA 학습까지 ComfyUI 안에서 하는 노드가 없다.
   * ASIS: 폴더 이미지 읽기, 자주 나오는 태그 걸러 내기, sd-scripts 학습을 팩 밖에서 한다.
   * TOBE: `LoadImagesFromDir`, `FilterFrequentTags`, `TrainSdxlLora` 노드로 한 워크플로에서 한다.
2. `SaveImageWithText`가 원본 파일 이름을 유지하지 못한다.
   * ASIS: 항상 번호로 저장하고, 다시 실행하면 같은 이미지가 또 저장된다.
   * TOBE: `name`을 주면 원본 이름으로 저장하고, 이미 있는 쌍은 건너뛴다.

### 원인 분석
1. `AlcheminePack/Lora`에는 이미지 다운로드와 이미지·캡션 저장 노드만 있다.
2. `SaveImageWithText`는 파일 이름을 `prefix`와 번호로만 정해, 같은 원본을 알아볼 방법이 없다.

### 솔루션
1. 데이터셋 폴더를 읽고, 캡션을 다듬고, 학습을 돌리는 노드 3개를 더한다.
   - `LoadImagesFromDir`: output 폴더 기준 폴더의 이미지를 파일 이름과 함께 리스트로 읽고, 파일이 바뀌면 다시 실행한다.
   - `FilterFrequentTags`: 전체 캡션 중 `threshold` 비율 이상에 나오는 태그를 지운다(`fixed_tags`는 남김).
   - `TrainSdxlLora`: `<ComfyUI>/sd-scripts`의 kohya sd-scripts로 `output/train/<name>/<version>/` 데이터셋을 학습하고, 노드에 경로와 loss 차트를 보여 준다(`web/js/train_sdxl_lora.js`).
2. `SaveImageWithText`가 원본 이름을 이어받아, 같은 이미지를 두 번 저장하지 않게 한다.
   - 선택 입력 `name`, `overwrite`를 더하고, 출력 노드로 바꾼다.
3. 노드 추가이므로 version을 8.1.0으로 올리고, `README.md`, `README_ko.md`에 노드 설명을 더한다.

### 테스트
테스트: [`tests/issue/51-add-lora-dataset-and-sdxl-lora-training-nodes/test_lora_training_nodes.py`](../../../tests/issue/51-add-lora-dataset-and-sdxl-lora-training-nodes/test_lora_training_nodes.py)

| 검증 내용 | 개선점 | Before | After |
|---|---|---|---|
| 새 노드 3개가 등록되어 있다 | 워크플로에서 쓸 수 있다 | 실패: 없다 | 통과 |
| `SaveImageWithText`에 `name`, `overwrite` 입력이 있다 | 원본 이름을 유지한다 | 실패: 없다 | 통과 |
| README 두 파일에 새 노드 설명이 있다 | 사용법을 찾을 수 있다 | 실패: 없다 | 통과 |

새 이슈 테스트 6개 중 0개(0%) → 6개(100%) 통과, 노드별 테스트 37개를 더해 합계 114 → 157 passed.

실행 방법:

```bash
uv venv
uv pip install --python .venv/bin/python -r tests/requirements.txt
.venv/bin/python -m pytest -c tests/pytest.ini tests
```

### References
- https://github.com/alchemine/comfyui-alchemine-pack/issues/51
