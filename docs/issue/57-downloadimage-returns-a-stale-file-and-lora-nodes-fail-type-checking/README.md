# #57 DownloadImage returns a stale file and LoRA nodes fail type checking

이슈: https://github.com/alchemine/comfyui-alchemine-pack/issues/57

### 이슈
1. `DownloadImage`가 새 URL에 옛 이미지를 돌려준다.
   * ASIS: `1.png`, `2.png`, `3.png`에서 `2.png`를 지우고 새 URL을 받으면 `3.png`(옛 이미지)가 나온다.
   * TOBE: 새 URL의 이미지가 나온다.
2. 같은 URL을 다시 받는다.
   * ASIS: 캐시가 비면(서버 재시작 등) 같은 URL을 새 번호로 다시 내려받는다.
   * TOBE: 이미 받은 파일을 그대로 쓴다.
3. 쿼리가 있는 URL의 확장자가 깨진다.
   * ASIS: `http://host/a.jpg?w=512` → `1.jpg?w=512`
   * TOBE: `.jpg`
4. `nodes/lora.py`에서 Pylance 타입 에러가 21개 나온다.
   * ASIS: `dir_path: str`에 `Path`를 다시 넣어 `/`, `.glob` 등이 에러로 잡힌다.
   * TOBE: 0개.

### 원인 분석
1. 파일 번호를 `1 + 폴더의 파일 수`로 정해, 파일을 지우면 남은 파일과 번호가 겹친다. 파일이 있으면 다운로드를 건너뛴다.
2. 파일 이름이 URL과 상관없는 번호라, 이미 받은 URL인지 알 수 없다.
3. 확장자를 URL 전체의 마지막 점 뒤(`url.split(".")[-1]`)로 정해 쿼리와 fragment가 섞인다.
4. 타입 표기가 실제 값과 다르다.
   - 인자(`dir_path: str`, `threshold: list[float]`)에 다른 타입의 값을 다시 넣는다.
   - 반환형이 실제 튜플 길이와 다르다(`IS_CHANGED -> str`, `SaveImageWithText -> tuple[str]`).
   - `fixed_tags: list[str] = None`에 `None`이 빠져 있고, `proc.stdout`을 `None`일 수 있다고 본다.

### 솔루션
1. 다운로드한 파일 이름을 URL 해시로 정해, 같은 URL은 같은 파일, 다른 URL은 다른 파일이 된다.
   - `sha256(url)` 앞 16자를 이름으로 쓴다. 파일이 있으면 내려받지 않는다.
   - 기본 캐시와 같은 일을 하던 `DownloadImage.IS_CHANGED`를 지운다.
2. 확장자를 URL 경로에서만 가져온다.
   - `Path(urlparse(url).path).suffix`
3. 타입 표기를 실제 값에 맞춘다.
   - 인자에 다시 넣던 값은 새 이름(`folder`, `ratio`)으로 받는다.
   - 반환형을 `tuple[str, str]`로, `fixed_tags`를 `list[str] | None`으로 고친다.
4. `#13` 테스트의 기대 파일 이름(`downloads/1.png`)을 해시 이름에 맞춘다.

### 테스트
테스트: [`tests/issue/57-downloadimage-returns-a-stale-file-and-lora-nodes-fail-type-checking/`](../../../tests/issue/57-downloadimage-returns-a-stale-file-and-lora-nodes-fail-type-checking/)

| 검증 내용 | 개선점 | Before | After |
|---|---|---|---|
| 저장 파일의 확장자가 URL 경로의 것이다 (10건) | 쿼리, fragment가 확장자에 섞이지 않는다 | 5/10 통과 | 10/10 통과 |
| 파일을 지운 뒤 새 URL을 받으면 그 URL의 이미지가 나온다 (3건) | 옛 이미지를 돌려주지 않는다 | 0/3 통과 | 3/3 통과 |
| 같은 URL을 두 번 받으면 한 번만 내려받는다 (4건) | 중복 파일이 쌓이지 않는다 | 0/4 통과 | 4/4 통과 |
| 대표 예제 5건 (사용자 지적 3건, 직접 추가 2건) | 이슈 1~3이 해결된다 | 0/5 통과 | 5/5 통과 |
| Pylance 진단 (`nodes/lora.py`) | 타입 검사를 통과한다 | 21개 | 0개 |

새 이슈 테스트 22개 중 5개(23%) → 22개(100%) 통과, 통과율 +77%p. 합계 205 passed(기존 183 + 새 22, #13 테스트는 기대 이름만 고침). Pylance 진단 21개 → 0개.

실행 방법:

```bash
uv venv
uv pip install --python .venv/bin/python -r tests/requirements.txt
.venv/bin/python -m pytest -c tests/pytest.ini tests
```

### References
- https://github.com/alchemine/comfyui-alchemine-pack/issues/57
