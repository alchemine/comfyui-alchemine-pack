# #21 Add FilterColors node

이슈: https://github.com/alchemine/comfyui-alchemine-pack/issues/21

## 이슈
- `red dress, blue dress`처럼 같은 대상에 색이 여러 개 붙은 프롬프트를 정리하는 노드가 없다.
- `FilterTags`에 `<color> dress`를 넣으면 둘 다 지워진다. `FilterSubtags`는 색 중복을 보지 않는다.

## 해결책
- `FilterColors` 노드를 `AlcheminePack/Prompt`에 추가한다.
- "색 + 명사" 모양의 태그에서 색 뒤의 부분이 같은 태그가 다시 나오면 나중 것을 지운다. `black hair, black dress`는 대상이 달라서 둘 다 남는다.
- 색 목록은 `resources/wildcards.yaml`의 `color` 키를 쓴다. `FilterTags`의 `<color>`와 같은 목록이다.
- 긴 색 이름부터 맞춰 본다. `light blue dress`의 대상은 `dress`다.
- 가중치는 떼고 비교한다. `BREAK`는 자리와 공백을 지키고, 중복은 프롬프트 전체를 기준으로 본다.
- 출력은 `processed_text`, `filtered_tags`다.
- 태그를 걸러 내는 공통 부분은 `BasePrompt.drop_tags`에 둔다.

## 테스트 계획
테스트는 `tests/issue/21-add-filtercolors-node/`에 있다. 노드의 `execute`를 직접 호출한다.

| 테스트 | 무엇을 테스트하는가 | 기대 결과 |
|---|---|---|
| `test_the_first_colour_of_a_thing_stays` | `1girl, red dress, blue dress, white shirt, black shirt` | `1girl, red dress, white shirt`, 지워진 태그는 `blue dress, black shirt` |
| `test_different_things_keep_their_colours` | `black hair, black dress, blue eyes, long hair` | 입력 그대로 |
| `test_a_two_word_colour_is_one_colour` | `light blue dress, red dress` | `light blue dress`만 남는다 |
| `test_weights_do_not_hide_a_colour` | `(red dress:1.2), (blue dress)` | `(red dress:1.2)`만 남는다 |
| `test_break_keeps_its_place_and_the_groups_are_one_picture` | `red dress, smile BREAK blue dress, sky` | `red dress, smile BREAK sky` |

실행 방법:

```bash
uv venv
uv pip install --python .venv/bin/python -r requirements.txt -r tests/requirements.txt
.venv/bin/python -m pytest -c tests/pytest.ini tests
```

## 테스트 결과
| | 수정 전 (`2ae6595`) | 수정 후 |
|---|---|---|
| 이 이슈의 테스트 5개 | 실패 (`FilterColors`가 없다) | 통과 |
| 기존 테스트 23개 | 통과 | 통과 |
| 합계 | 5 failed, 23 passed | 28 passed |
