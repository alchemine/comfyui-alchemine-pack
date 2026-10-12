# #48 Remove the Adjust Image node moved to comfyui-filter-pack

이슈: https://github.com/alchemine/comfyui-alchemine-pack/issues/48

### 이슈
1. `AdjustImage` 노드를 [comfyui-filter-pack](https://github.com/alchemine/comfyui-filter-pack)으로 옮긴다.
   * ASIS: `AdjustImage`가 이 팩의 `AlcheminePack/Image`에 등록되어 있다.
   * TOBE: 이 팩에서 `AdjustImage`를 지우고, README의 Related Packs 표에 comfyui-filter-pack을 더한다.

### 원인 분석
1. 같은 노드 id가 두 팩에 있으면, 두 팩을 함께 설치했을 때 한쪽이 다른 쪽을 덮어쓴다.

### 솔루션
1. 이 팩에서 `AdjustImage`를 지워, 두 팩을 함께 설치해도 노드가 하나만 등록되게 한다.
   - `nodes/image.py`를 지우고, `__init__.py`의 import와 등록 두 곳을 지운다.
   - `README.md`, `README_ko.md`의 Image Nodes 절과 첫 문단의 "image adjustment"를 지운다.
   - `pyproject.toml`의 description에서 "Adjust Image"를 지운다.
2. 옮겨 간 노드를 찾을 수 있게 README의 Related Packs 표에 comfyui-filter-pack 행을 더한다.
   - `README.md`, `README_ko.md` 둘 다.
3. 노드 제거는 호환 불가 변경이므로 version을 8.0.0으로 올린다.

### 테스트
테스트: [`tests/issue/48-remove-the-adjust-image-node-moved-to-comfyui-filter-pack/test_removed_adjust_image.py`](../../../tests/issue/48-remove-the-adjust-image-node-moved-to-comfyui-filter-pack/test_removed_adjust_image.py)

| 검증 내용 | 개선점 | Before | After |
|---|---|---|---|
| `NODE_CLASS_MAPPINGS`에 `AdjustImage`가 없다 | 노드 id가 두 팩에 겹치지 않는다 | 실패: 등록되어 있다 | 통과 |
| `nodes/image.py`가 없다 | 쓰지 않는 코드가 남지 않는다 | 실패: 있다 | 통과 |
| README 두 파일에 팩 6개의 GitHub 링크가 있다 | 옮겨 간 노드를 찾을 수 있다 | 실패: filter pack 링크가 없다 | 통과 |
| README 두 파일에 `AdjustImage` 설명이 없다 | 없는 노드를 설명하지 않는다 | 실패: 있다 | 통과 |

새 테스트 6개 중 0개(0%) → 6개(100%) 통과, 기존 108개는 그대로 통과(합계 114 passed).

실행 방법:

```bash
uv venv
uv pip install --python .venv/bin/python -r tests/requirements.txt
.venv/bin/python -m pytest -c tests/pytest.ini tests
```

### References
- https://github.com/alchemine/comfyui-alchemine-pack/issues/48
- https://github.com/alchemine/comfyui-filter-pack
