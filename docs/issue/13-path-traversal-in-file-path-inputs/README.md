# #13 Path traversal in file path inputs

이슈: https://github.com/alchemine/comfyui-alchemine-pack/issues/13

## 이슈

사용자가 입력한 문자열을 검증 없이 경로의 일부로 사용해서, 기준 폴더 밖의
파일을 쓰거나 읽을 수 있다.

| 위치 | 입력 | 결과 |
|---|---|---|
| `nodes/lora.py` `DownloadImage` | `dir_path` | 출력 폴더 밖에 파일 쓰기 |
| `nodes/lora.py` `SaveImageWithText` | `dir_path`, `prefix` | 출력 폴더 밖에 `.txt`, `.png` 쓰기 |
| `nodes/api.py` `LoadWorkflow` | `filename` | `workflows` 폴더 밖의 파일 읽기 |

- `dir_path`에 `../escaped`를 넣으면 출력 폴더의 부모에 폴더가 생기고 그 안에
  파일이 저장된다.
- `prefix`에 `../../escaped/p`를 넣으면 `dir_path`가 정상이어도 밖에 저장된다.
- `filename`은 콤보 입력이지만 API로 보낸 워크플로에는 목록에 없는 값이 올 수
  있다. `../../../secret.txt`를 넣으면 그 파일의 내용이 출력으로 나온다.

원인은 세 곳 모두 같다. 경로를 조합한 뒤 최종 경로가 기준 폴더 안에 있는지
확인하지 않는다.

`nodes/grok.py`는 `folder_paths.get_save_image_path`로 경로를 만들고, 이 함수가
출력 폴더를 벗어나는 경로를 거부하므로 해당하지 않는다.

## 해결책

최종 경로가 기준 폴더 안에 있는지 확인하는 함수 하나를 `nodes/lib/utils.py`에
두고, 세 곳에서 폴더를 만들거나 파일을 열기 전에 호출한다.

- `os.path.realpath`로 두 경로를 푼 뒤 `os.path.commonpath`로 비교한다.
  심볼릭 링크로 빠져나가는 경우도 함께 막힌다.
- 폴더가 아니라 최종 파일 경로를 검사한다. `dir_path`와 `prefix`가 한 번의
  검사로 막힌다.
- 벗어나면 `ValueError`로 실패한다.
- `folder_paths.is_within_directory`는 쓰지 않는다. 2026-07에 추가된 함수라
  그보다 오래된 ComfyUI에서는 없다.

## 테스트 계획

테스트는 `tests/issue/13-path-traversal-in-file-path-inputs/`에 있다.
`folder_paths`의 출력 폴더와 사용자 폴더를 임시 폴더로 바꾸고 노드를 직접
호출한다. `DownloadImage`의 `requests.get`은 PNG를 돌려주는 가짜로 바꾼다.

| 테스트 | 무엇을 테스트하는가 | 기대 결과 |
|---|---|---|
| `test_download_image_writes_inside_output` | `dir_path="downloads"` | `output/downloads/1.png`가 생긴다 |
| `test_download_image_dir_path_cannot_leave_output` | `dir_path="../escaped_download"` | `ValueError`, 출력 폴더 밖에 아무것도 생기지 않는다 |
| `test_save_image_with_text_writes_inside_output` | `dir_path="saved"` | `output/saved/1.png`, `1.txt`가 생긴다 |
| `test_save_image_with_text_dir_path_cannot_leave_output` | `dir_path="../escaped_save"` | `ValueError`, 출력 폴더 밖에 아무것도 생기지 않는다 |
| `test_save_image_with_text_prefix_cannot_leave_output` | `prefix="../../escaped_prefix/p"` | `ValueError`, 밖의 폴더가 비어 있다 |
| `test_load_workflow_reads_inside_workflows` | `filename="sub/wf.json"` | 파일 내용을 돌려준다 |
| `test_load_workflow_filename_cannot_leave_workflows` | `filename="../../../secret.txt"` | `ValueError` |

실행 방법:

```bash
uv venv
uv pip install --python .venv/bin/python -r requirements.txt -r tests/requirements.txt
.venv/bin/python -m pytest -c tests/pytest.ini tests
```

## 테스트 결과

| 테스트 | 수정 전 (`cb1f9fe`) |
|---|---|
| `test_download_image_writes_inside_output` | 통과 |
| `test_download_image_dir_path_cannot_leave_output` | 실패: `ValueError`가 나지 않는다 |
| `test_save_image_with_text_writes_inside_output` | 통과 |
| `test_save_image_with_text_dir_path_cannot_leave_output` | 실패: `ValueError`가 나지 않는다 |
| `test_save_image_with_text_prefix_cannot_leave_output` | 실패: `ValueError`가 나지 않는다 |
| `test_load_workflow_reads_inside_workflows` | 통과 |
| `test_load_workflow_filename_cannot_leave_workflows` | 실패: `ValueError`가 나지 않는다 |
| 합계 | 4 failed, 3 passed |
