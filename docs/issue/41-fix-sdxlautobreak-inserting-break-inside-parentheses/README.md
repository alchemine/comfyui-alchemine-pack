# #41 SDXLAutoBreak inserts BREAK inside parentheses

이슈: https://github.com/alchemine/comfyui-alchemine-pack/issues/41

## 이슈
`SDXLAutoBreak`가 괄호 안의 쉼표 위치에 `BREAK`를 넣을 수 있다.

| 입력 | 결과 |
|---|---|
| `..., (@ @,:-1.1) (light particles,:-1.2) posing` | `..., (@ @` `BREAK` `:-1.1) (light particles,:-1.2) posing` |

괄호가 짝을 잃어서 가중치 문법이 깨진다.

- `:-1.1)` 같은 글자가 그대로 프롬프트로 들어간다.
- 지정한 가중치가 적용되지 않는다.

원인은 `split()`이 `re.finditer(r"[^,]+", seg)`로 단어를 나누는 것이다.
이 정규식은 괄호를 보지 않으므로, 괄호 안의 쉼표도 `BREAK`를 넣을 위치가 된다.

## 해결책
- 단어를 나눌 때 `BasePrompt.split_tags`를 쓴다.
- `split_tags`는 짝을 이루는 괄호 안의 쉼표에서 나누지 않는다.
- 이스케이프된 괄호와 짝이 없는 괄호는 글자로 본다. (#33)

## 테스트 계획
테스트는 `tests/issue/41-fix-sdxlautobreak-inserting-break-inside-parentheses/`에 있다.
가짜 `clip`은 단어와 문장 부호를 토큰 하나씩으로 센다.

| 테스트 | 무엇을 테스트하는가 | 기대 결과 |
|---|---|---|
| `test_break_stays_out_of_parentheses` (12개) | 앞에 붙이는 태그 수를 20~31로 바꿔 75토큰 경계를 괄호 그룹 곳곳에 둔다 | `BREAK`로 나눈 모든 구간에서 `(`와 `)`의 수가 같다 |
| `test_break_still_splits_between_tags` | 괄호가 없는 긴 프롬프트 | `BREAK`가 한 번 들어가고, 빼면 원문과 같다 |

실행 방법:

```bash
uv venv
uv pip install --python .venv/bin/python -r requirements.txt -r tests/requirements.txt
.venv/bin/python -m pytest -c tests/pytest.ini tests
```
