# #23 Add FilterPlurals node

이슈: https://github.com/alchemine/comfyui-alchemine-pack/issues/23

## 이슈
- `arm up, arms up`처럼 단수형과 복수형만 다른 태그가 함께 있는 프롬프트를 정리하는 노드가 없다.
- `FilterSubtags`는 `arm up`을 `arms up`의 부분 태그로 보지 않아서 둘 다 남긴다.

## 해결책
- `FilterPlurals` 노드를 `AlcheminePack/Prompt`에 추가한다.
- 단어 끝의 `s`만 다른 두 태그가 있으면 나중 것을 지운다. `s`는 태그의 어느 단어에 있어도 된다 (`hand on hip`과 `hands on hips`).
- 4글자 이상이고 `s`로 끝나는 단어에서만 `s`를 떼어 비교한다. `ass`, `abs`는 건드리지 않고, `glasses`는 `glass`가 되지 않는다.
- 똑같은 태그가 두 번 나온 것은 복수형 문제가 아니므로 그대로 둔다.
- 가중치는 떼고 비교한다. `BREAK`는 자리와 공백을 지키고, 중복은 프롬프트 전체를 기준으로 본다.
- 출력은 `processed_text`, `filtered_tags`다. 걸러 내는 부분은 `BasePrompt.drop_tags`를 쓴다.

## 테스트 계획
테스트는 `tests/issue/23-add-filterplurals-node/`에 있다. 노드의 `execute`를 직접 호출한다.

| 테스트 | 무엇을 테스트하는가 | 기대 결과 |
|---|---|---|
| `test_the_later_spelling_goes` | `arm up, arms up`과 그 반대 순서 | 어느 쪽이든 나중 것이 지워진다 |
| `test_any_word_of_the_tag_may_carry_the_s` | `hand on hip, hands on hips, boots, boot` | `hand on hip, boots` |
| `test_short_words_are_not_plurals` | `ass, as, abs, ab` | 입력 그대로 |
| `test_a_different_tag_is_not_a_plural` | `glass, glasses, dress, dresses, arm up, arm down` | 입력 그대로 |
| `test_the_same_tag_twice_is_not_this_filters_business` | `arms up, arms up` | 입력 그대로 |
| `test_weights_and_break` | `(arm up:1.2), smile BREAK arms up, sky` | `(arm up:1.2), smile BREAK sky` |

실행 방법:

```bash
uv venv
uv pip install --python .venv/bin/python -r requirements.txt -r tests/requirements.txt
.venv/bin/python -m pytest -c tests/pytest.ini tests
```

## 테스트 결과
| | 수정 전 (`c569b40`) | 수정 후 |
|---|---|---|
| 이 이슈의 테스트 6개 | 실패 (`FilterPlurals`가 없다) | 통과 |
| 기존 테스트 28개 | 통과 | 통과 |
| 합계 | 6 failed, 28 passed | 34 passed |
