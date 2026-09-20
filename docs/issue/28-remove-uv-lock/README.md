# #28 Remove uv.lock

이슈: https://github.com/alchemine/comfyui-alchemine-pack/issues/28

## 이슈
- `uv.lock`이 리포에 남아 있지만 쓰이지 않는다.
- 의존성은 `requirements.txt`와 `tests/requirements.txt`로 관리하고, 설치는 `uv pip install -r`로 한다. 이 방식은 `uv.lock`을 읽지도 만들지도 않는다.
- 리포 안에 `uv lock`, `uv sync`, `uv run`을 쓰는 곳이 없다.
- 파일에 적힌 팩 버전이 `2.1.0`으로, 현재 버전과 맞지 않는다.

## 해결책
- `uv.lock`을 지운다.
- `.gitignore`에는 추가하지 않는다. 이 파일을 만드는 절차가 없으므로, 다시 나타나면 그것이 고쳐야 할 신호다.

## 테스트 계획
| 테스트 | 기대 결과 |
|---|---|
| `git ls-files uv.lock` | 아무것도 출력하지 않는다 |
| `git grep -n "uv.lock\|uv sync\|uv run\|uv lock"` | 이 문서 말고는 나오지 않는다 |
| `.venv/bin/python -m pytest -c tests/pytest.ini tests` | 기존 테스트가 그대로 통과한다 |

## 테스트 결과
| 테스트 | 수정 전 (`1facead`) | 수정 후 |
|---|---|---|
| `git ls-files uv.lock` | `uv.lock` | 출력 없음 |
| `git grep` | 출력 없음 | 이 문서만 나옴 |
| `pytest` | 38 passed | 38 passed |
