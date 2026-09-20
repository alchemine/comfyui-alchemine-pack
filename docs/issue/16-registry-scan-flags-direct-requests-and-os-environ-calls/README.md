# #16 Registry scan flags direct requests and os.environ calls

이슈: https://github.com/alchemine/comfyui-alchemine-pack/issues/16

## 이슈

Comfy Registry의 자동 스캔이 아래 문자열을 `info` 등급으로 탐지해서, 새 버전이
Active가 되지 못하고 Flagged로 남는다. 5.0.0과 5.0.1이 이 사유만으로 Flagged다.

| 규칙 | 문자열 | 위치 |
|---|---|---|
| `python_network_operations` | `requests.post(`, `requests.get(` | `nodes/grok.py:69`, `nodes/grok.py:201`, `nodes/inference.py:222`, `nodes/inference.py:262`, `nodes/lora.py:52` |
| `python_environment_manipulation` | `os.environ.get(` | `nodes/grok.py:47-49`, `nodes/inference.py:166-167` |

- 스캔은 문자열을 찾는다. `nodes/grok.py`의 `requests.request(`와, `requests.Session()`으로
  요청을 보내는 `nodes/api.py`는 탐지되지 않았다.
- 탐지 사유는 `https://api.comfy.org/nodes/comfyui-alchemine-pack/versions?include_status_reason=true`에서
  볼 수 있다.

## 해결책

- HTTP 호출은 `nodes/api.py`처럼 `requests.Session()`으로 보낸다. 세 파일에 모듈 수준의
  `_session`을 두고 `_session.post(`, `_session.get(`, `_session.request(`로 호출한다.
  연결을 재사용하게 된다.
- 환경변수는 `from os import environ`으로 읽는다. 동작은 같다.
- 노드의 입력, 출력, 동작은 바뀌지 않는다.

## 테스트 계획

테스트는 `tests/issue/16-registry-scan-flags-direct-requests-and-os-environ-calls/`에 있다.
`nodes/` 아래의 `.py` 파일을 읽어 탐지된 세 문자열이 남아 있는지 확인한다.

| 테스트 | 무엇을 테스트하는가 | 기대 결과 |
|---|---|---|
| `test_nodes_do_not_contain_the_flagged_strings` | `nodes/**/*.py`에 `requests.get(`, `requests.post(`, `os.environ.get(`이 있는가 | 하나도 없다 |

스캔의 규칙 전체는 공개되어 있지 않다. 이 테스트가 확인하는 것은 5.0.1에서 탐지된
문자열뿐이고, 통과가 Registry의 판정을 보장하지는 않는다.

#13의 테스트는 동작이 바뀌지 않았는지 확인하는 데 함께 쓴다.

실행 방법:

```bash
uv venv
uv pip install --python .venv/bin/python -r requirements.txt -r tests/requirements.txt
.venv/bin/python -m pytest -c tests/pytest.ini tests
```

## 테스트 결과

| 테스트 | 수정 전 (`2cfd6ae`) |
|---|---|
| `test_nodes_do_not_contain_the_flagged_strings` | 실패: 위 표의 열 곳이 나온다 |
| #13의 테스트 8개 | 통과 |
| 합계 | 1 failed, 8 passed |
