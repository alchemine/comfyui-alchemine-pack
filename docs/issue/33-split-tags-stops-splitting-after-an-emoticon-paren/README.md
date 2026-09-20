# #33 split_tags stops splitting after an emoticon paren

이슈: https://github.com/alchemine/comfyui-alchemine-pack/issues/33

## 이슈
`BasePrompt.split_tags`가 `>:(`나 `:)` 같은 이모티콘 태그를 만나면 그 뒤의 쉼표를 나누지 못한다.

| 입력 | 결과 | 기대 |
|---|---|---|
| `>:(, (smile:1.2), sky` | 조각 1개 | 조각 3개 |
| `:), (smile:1.2), sky` | 조각 1개 | 조각 3개 |
| `pozyomka \(arknights, moon` | 조각 1개 | 조각 2개 |

`RemoveWeights`가 이 함수로 태그를 나누므로, `>:(, (smile:1.2), sky`의 가중치를 떼지 못하고 입력을 그대로 돌려준다.

원인은 괄호의 깊이를 세면서 모든 `(`와 `)`를 강조 괄호로 보는 것이다.

- 짝이 없는 `(`는 깊이를 1로 올린 채 끝까지 간다.
- 짝이 없는 `)`는 깊이를 -1로 내린다.
- 어느 쪽이든 "깊이가 0일 때만 나눈다"는 조건이 다시는 참이 되지 않는다.

## 해결책
- 짝을 이루는 `(`와 `)`만 강조 괄호로 본다. 스택으로 짝을 찾고, 그 짝 안에 있는 쉼표만 나누지 않는다.
- `\(`와 `\)`처럼 이스케이프된 괄호는 글자로 본다.
- 짝이 없는 괄호도 글자로 본다. `>:(`의 `(`, `:)`의 `)`가 여기에 해당한다.

한계가 하나 있다. `>:(, smile, :)`처럼 여는 이모티콘과 닫는 이모티콘이 한 프롬프트에 함께 있으면 둘이 짝을 이뤄 강조 그룹 하나로 읽힌다. 프롬프트 문법 자체가 그렇게 읽고 ComfyUI의 해석도 같으므로, 이 함수가 달리 판단할 근거가 없다. 구분하려면 `>:\(`, `:\)`처럼 이스케이프해야 한다.

## 테스트 계획
테스트는 `tests/issue/33-split-tags-stops-splitting-after-an-emoticon-paren/`에 있다.

| 테스트 | 무엇을 테스트하는가 | 기대 결과 |
|---|---|---|
| `test_split_tags` (8개) | 강조 그룹 안의 쉼표, 겹친 그룹, `>:(`, `:)`, 그룹 안의 `:)`, 이스케이프된 괄호, 그룹 안의 이스케이프된 괄호, 닫히지 않은 이스케이프 | 강조 그룹의 쉼표만 남고 나머지 쉼표에서 나뉜다 |
| `test_remove_weights_after_an_emoticon` | `>:(, (smile:1.2),\n[sky]`와 `:), (smile:1.2),\n[sky]` | 이모티콘은 그대로, 뒤의 가중치는 빠진다 |
| `test_an_opener_and_a_closer_pair_up_even_when_both_are_emoticons` | `>:(, smile, :), sky`와 이스케이프한 같은 입력 | 앞의 것은 그룹 하나와 `sky`, 뒤의 것은 조각 4개 |

실행 방법:

```bash
uv venv
uv pip install --python .venv/bin/python -r requirements.txt -r tests/requirements.txt
.venv/bin/python -m pytest -c tests/pytest.ini tests
```

## 테스트 결과
| | 수정 전 (`a0cac00`) | 수정 후 |
|---|---|---|
| `test_split_tags` | `>:(`, `:)`, 닫히지 않은 이스케이프의 3개 실패 | 8개 모두 통과 |
| `test_remove_weights_after_an_emoticon` | 실패 | 통과 |
| `test_an_opener_and_a_closer_pair_up_even_when_both_are_emoticons` | 실패 (이스케이프한 입력이 조각 1개가 된다) | 통과 |
| 기존 테스트 66개 | 통과 | 통과 |
| 합계 | 5 failed, 71 passed | 76 passed |
