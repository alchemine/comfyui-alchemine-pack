# #55 Load dataset folders in chunks and keep dotted file stems in SaveImageWithText

이슈: https://github.com/alchemine/comfyui-alchemine-pack/issues/55

### 이슈
1. `LoadImagesFromDir`가 큰 데이터셋 폴더를 나눠 읽지 못한다.
   * ASIS: 폴더의 이미지를 항상 전부 한 번에 읽는다.
   * TOBE: `count=2, chunk=1`이면 정렬된 이미지 중 3, 4번째만 읽는다.
2. `SaveImageWithText`가 점이 있는 이름을 잘라서 저장한다.
   * ASIS: `name="2024-01-01T00.00.00.000Z_1.jpg"` → `2024-01-01T00.00.00.png`
   * TOBE: `2024-01-01T00.00.00.000Z_1.png`

### 원인 분석
1. `LoadImagesFromDir`에 읽을 범위를 정하는 입력이 없다.
2. `Path.with_suffix`는 마지막 점 뒤를 확장자로 보고 바꾼다. `Path(name).stem`에 점이 남아 있으면 그 뒤가 잘린다.

### 솔루션
1. 폴더를 `count`장씩 나눠, `chunk`번째 묶음만 읽게 한다.
   - 선택 입력 `count`(0이면 전체), `chunk`(`control_after_generate`)를 더한다.
   - `chunk_paths`로 정렬된 목록을 자르고, `IS_CHANGED`도 그 묶음의 파일만 본다.
2. 확장자를 바꾸지 않고 이름 뒤에 붙여, 이름 전체를 유지한다.
   - `with_suffix` 대신 `f"{prefix_path.name}.txt"`, `.png`로 경로를 만든다.
3. `README.md`, `README_ko.md`의 `LoadImagesFromDir` 표에 `count`, `chunk`를 더한다.

### 테스트
테스트: [`tests/issue/55-load-dataset-folders-in-chunks-and-keep-dotted-file-stems-in-saveimagewithtext/`](../../../tests/issue/55-load-dataset-folders-in-chunks-and-keep-dotted-file-stems-in-saveimagewithtext/)

| 검증 내용 | 개선점 | Before | After |
|---|---|---|---|
| 묶음 크기와 번호에 따라 정렬된 이미지의 일부만 고른다 (10건) | 큰 폴더를 나눠 읽는다 | 0/10 통과 | 10/10 통과 |
| `chunk`를 올리며 실행하면 모든 이미지를 한 번씩 읽는다 | 묶음 사이에 빠지거나 겹치는 파일이 없다 | 실패 | 통과 |
| 점이 있는 이름 10개를 저장해도 이름 전체가 남는다 | 원본 이름을 잃지 않는다 | 2/10 통과 | 10/10 통과 |
| 대표 예제 5건 (타임스탬프 이름 3건, 묶음 읽기 2건) | 이슈의 두 문제가 해결된다 | 0/5 통과 | 5/5 통과 |

새 이슈 테스트 26개 중 2개(8%) → 26개(100%) 통과, 통과율 +92%p. 합계 159 passed / 24 failed → 183 passed.

실행 방법:

```bash
uv venv
uv pip install --python .venv/bin/python -r tests/requirements.txt
.venv/bin/python -m pytest -c tests/pytest.ini tests
```

### References
- https://github.com/alchemine/comfyui-alchemine-pack/issues/55
