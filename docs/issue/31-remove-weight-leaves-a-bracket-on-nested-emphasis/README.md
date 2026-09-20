# #31 remove_weight leaves a bracket on nested emphasis

이슈: https://github.com/alchemine/comfyui-alchemine-pack/issues/31

## 이슈
`BasePrompt.remove_weight`와 `BasePrompt.normalize_tag`가 겹친 강조 괄호에서 닫는 괄호를 하나 남긴다.

| 입력 | 결과 | 기대 |
|---|---|---|
| `((cat))` | `cat)` | `cat` |
| `[[cat]]` | `cat]` | `cat` |
| `((star \(sky\)))` | `star \(sky\))` | `star \(sky\)` |

- `RemoveWeights`가 `((cat)), [[dog]]`을 `cat), dog]`으로 만든다.
- 두 함수의 docstring은 `((cat))`과 `[[cat]]`을 지원한다고 적고 있다.
- `FilterTags`와 `FilterSubtags`는 영향을 받지 않는다. 비교하기 전에 `standardize_prompt`가 `((bad))`를 `(bad:1.21)`로 바꾸기 때문이다.

원인은 정규식 `^([\(\[]+)(.+)([\)\]]+)$`다. 가운데 `(.+)`가 탐욕적이어서 닫는 괄호를 마지막 하나만 남기고 모두 가져간다.

## 해결책
- `BasePrompt.unwrap`을 추가한다. 여는 괄호의 수만큼만 닫는 괄호를 떼어 낸다.
- 정규식을 `(.+?)`로 바꾸는 것으로는 풀리지 않는다. `(star \(sky\))`의 끝에 있는 글자 괄호 `\)`까지 떼어 내게 된다. 개수를 세야 한다.
- `remove_weight`와 `normalize_tag`의 괄호 분기를 `unwrap` 호출로 바꾼다. 두 함수에 같은 정규식이 따로 들어 있었다.

## 테스트 계획
테스트는 `tests/issue/31-remove-weight-leaves-a-bracket-on-nested-emphasis/`에 있다.

| 테스트 | 무엇을 테스트하는가 | 기대 결과 |
|---|---|---|
| `test_the_tag_comes_out_of_its_brackets_whole` (12개) | `cat`, `(cat)`, `((cat))`, `(((cat)))`, `[cat]`, `[[cat]]`, `(cat:1.2)`, 글자 괄호가 든 태그 세 가지, `:)`, `(cat`을 두 함수에 넣는다 | 강조 괄호만 빠지고 태그는 온전하다. 강조가 아닌 것은 그대로다 |
| `test_remove_weights_on_nested_emphasis` | `((cat)), [[dog]], (bird)` | `cat, dog, bird` |

실행 방법:

```bash
uv venv
uv pip install --python .venv/bin/python -r requirements.txt -r tests/requirements.txt
.venv/bin/python -m pytest -c tests/pytest.ini tests
```

## 테스트 결과
| | 수정 전 (`3b3ea6e`) | 수정 후 |
|---|---|---|
| 이 이슈의 테스트 13개 | 5개 실패 (`((cat))`, `(((cat)))`, `[[cat]]`, `((star \(sky\)))`, `RemoveWeights`) | 통과 |
| 기존 테스트 53개 | 통과 | 통과 |
| 합계 | 5 failed, 61 passed | 66 passed |
