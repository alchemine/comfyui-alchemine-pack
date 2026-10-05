# #43 Remove the API nodes moved to comfyui-api-pack

이슈: https://github.com/alchemine/comfyui-alchemine-pack/issues/43

## 이슈
- 외부 API를 호출하는 노드 8개를 comfyui-api-pack(v1.0.0)으로 옮겼다.
  - 원격 ComfyUI API: `LoadWorkflow`, `ApiGenerate`, `ApiSubmit`, `ApiCollect`
  - xAI API: `GrokGenerate`, `GrokSubmit`, `GrokCollect`
  - OpenAI 호환 API: `OpenAIInference`
- 같은 노드 id가 두 팩에 있어서, 두 팩을 함께 설치하면 한쪽이 다른 쪽을 덮어쓴다.
- README는 분리된 팩의 안내가 절 사이 네 곳에 흩어져 있고, api pack 안내는 없다.

## 해결책
- `nodes/api.py`, `nodes/grok.py`, `nodes/inference.py`, `nodes/lib/joblock.py`와 `__init__.py`의 등록을 지운다.
- `.env`를 읽는 노드가 남지 않으므로 `.env.example`, `utils.py`의 `load_dotenv`, `requirements.txt`의 `python-dotenv`, README의 Configuration 절을 지운다.
- `.gitignore`의 `jobs.lock`, 그림 `workflows/...-workflow-API.png`, `...-workflow-Inference.png`를 지운다.
- #13의 `LoadWorkflow` 테스트 두 개를 지운다. 같은 테스트가 comfyui-api-pack에 있다.
- README 맨 위에 Related Packs 표(팩 5개와 링크)를 두고, 흩어진 안내 네 곳을 지운다. `README_ko.md`도 같다.
- `pyproject.toml`의 version을 7.0.0으로 올리고, description에서 Inference, API, Grok 줄을 지운다.

## 테스트 계획
테스트는 `tests/issue/43-remove-the-api-nodes-moved-to-comfyui-api-pack/`에 있다.

| 테스트 | 무엇을 테스트하는가 | 기대 결과 |
|---|---|---|
| `test_moved_node_ids_are_not_registered` | `__init__.py`의 `NODE_CLASS_MAPPINGS` 키에 옮긴 id 8개가 있는가 | 하나도 없다 |
| `test_moved_node_files_are_gone` | `api.py`, `grok.py`, `inference.py`, `joblock.py`, `.env.example`이 있는가 | 없다 |
| `test_dotenv_is_not_required` | `requirements.txt`에 `python-dotenv`가 있는가 | 없다 |
| `test_readme_links_every_other_pack` (2개) | `README.md`, `README_ko.md`에 팩 5개의 GitHub 링크가 있는가 | 모두 있다 |

실행 방법:

```bash
uv venv
uv pip install --python .venv/bin/python -r requirements.txt -r tests/requirements.txt
.venv/bin/python -m pytest -c tests/pytest.ini tests
```

## 테스트 결과
| 테스트 | 수정 전 (`bd9bebd`) | 수정 후 |
|---|---|---|
| `test_moved_node_ids_are_not_registered` | 실패: 8개 모두 등록되어 있다 | |
| `test_moved_node_files_are_gone` | 실패: 다섯 파일 모두 있다 | |
| `test_dotenv_is_not_required` | 실패: `python-dotenv>=1.0.0` | |
| `test_readme_links_every_other_pack` (2개) | 실패: api pack 링크가 없다 | |
| 기존 테스트 92개 | 통과 | |
| 합계 | 5 failed, 92 passed | |
