# #27 Prompt nodes drop the user's line breaks

이슈: https://github.com/alchemine/comfyui-alchemine-pack/issues/27

## 이슈
여러 줄로 쓴 프롬프트가 프롬프트 노드를 지나면 줄바꿈을 잃는다.

| 노드 | 증상 |
|---|---|
| `FilterColors`, `FilterPlurals`, `BoySubjectFilter`, `RemoveWeights` | 거를 것이 없어도 줄바꿈이 전부 `, `로 바뀐다. |
| `FilterTags`, `FilterSubtags` | 줄의 첫 태그가 지워지면 그 줄의 줄바꿈이 사라진다. `1girl, smile,\nbad, sky`에서 `bad`를 지우면 `1girl, smile, sky`가 된다. |
| `ProcessTags` | `filter_colors`나 `filter_plurals`가 켜져 있으면 위 증상을 그대로 물려받는다. |

`BoySubjectFilter`는 `BREAK`도 나누지 않는다. `1girl, solo BREAK sex`의 `solo BREAK sex`를 태그 하나로 읽어서 아무것도 탐지하지 못한다.

원인은 하나다. 줄바꿈을 뒤에 오는 태그의 앞 공백으로 취급한다.

- `drop_tags`, `BoySubjectFilter`, `RemoveWeights`는 태그의 공백을 떼고 `", "`로 다시 이어 붙인다. 원래의 공백은 어디에도 남지 않는다.
- `FilterTags`, `FilterSubtags`는 원문 조각을 `","`로 이어서 공백을 지키지만, 조각을 지우면 그 조각의 앞 공백인 줄바꿈도 함께 지워진다.

## 해결책
쉼표로 나눈 원문 조각을 다시 이어 붙이는 일을 `BasePrompt`의 함수 하나에 모은다.

- 남는 조각은 앞뒤 공백까지 원문 그대로 둔다.
- 지워지는 조각의 앞 공백에 줄바꿈이 있으면, 다음에 남는 조각이 그 공백을 물려받는다. 그 조각이 이미 줄바꿈으로 시작하면 자기 것을 쓴다.
- `drop_tags`(`FilterColors`, `FilterPlurals`), `FilterTags`, `FilterSubtags`가 이 함수로 조각을 잇는다.
- `BoySubjectFilter`는 `drop_tags`로 `solo`를 지우고, 추가할 태그는 프롬프트의 끝에 붙인다. `drop_tags`가 `BREAK`를 나누므로 `BREAK`를 사이에 둔 태그도 읽는다.
- `RemoveWeights`는 조각의 알맹이만 바꾸고 앞뒤 공백은 그대로 둔다.

## 테스트 계획
테스트는 `tests/issue/27-prompt-nodes-drop-the-user-s-line-breaks/`에 있다. 노드의 `execute`를 직접 호출한다.

| 테스트 | 무엇을 테스트하는가 | 기대 결과 |
|---|---|---|
| `test_a_prompt_with_nothing_to_filter_comes_back_as_it_was` (5개) | 줄바꿈, 빈 줄, 공백 두 칸, `BREAK`가 있고 거를 것이 없는 프롬프트를 `FilterTags`, `FilterSubtags`, `FilterColors`, `FilterPlurals`, `BoySubjectFilter`에 넣는다 | 입력 그대로 |
| `test_process_tags_with_nothing_to_filter_comes_back_as_it_was` | 같은 프롬프트를 기본값의 `ProcessTags`에 넣는다 | 입력 그대로 |
| `test_the_line_break_outlives_the_first_tag_of_its_line` (4개) | 줄의 첫 태그가 지워지는 프롬프트를 `FilterTags`, `FilterSubtags`, `FilterColors`, `FilterPlurals`에 넣는다 | 줄바꿈이 다음 태그 앞에 남는다 |
| `test_a_tag_dropped_from_the_middle_of_a_line_takes_only_itself` | `red dress,\nsmile, blue dress, sky` | `red dress,\nsmile, sky` |
| `test_the_last_tag_of_a_prompt_leaves_no_trailing_comma` | `red dress, sky,\nblue dress` | `red dress, sky` |
| `test_remove_weights_keeps_the_layout` | 여러 줄, 공백 두 칸, `BREAK`가 있는 가중치 프롬프트 | 가중치만 빠지고 배치는 그대로 |
| `test_boy_subject_filter_keeps_the_layout` | `1girl, solo,\nsex, smile` | `1girl,\nsex, smile, ((1boy)), hetero` |
| `test_boy_subject_filter_reads_across_break` | `1girl, solo BREAK sex, smile` | `1girl BREAK sex, smile, ((1boy)), hetero` |

실행 방법:

```bash
uv venv
uv pip install --python .venv/bin/python -r requirements.txt -r tests/requirements.txt
.venv/bin/python -m pytest -c tests/pytest.ini tests
```

## 테스트 결과
| 테스트 | 수정 전 (`1facead`) |
|---|---|
| `test_a_prompt_with_nothing_to_filter_comes_back_as_it_was` | `FilterTags`, `FilterSubtags` 통과. `FilterColors`, `FilterPlurals`, `BoySubjectFilter` 실패 |
| `test_process_tags_with_nothing_to_filter_comes_back_as_it_was` | 실패 |
| `test_the_line_break_outlives_the_first_tag_of_its_line` | 4개 모두 실패 |
| `test_a_tag_dropped_from_the_middle_of_a_line_takes_only_itself` | 실패 |
| `test_the_last_tag_of_a_prompt_leaves_no_trailing_comma` | 통과 |
| `test_remove_weights_keeps_the_layout` | 실패 |
| `test_boy_subject_filter_keeps_the_layout` | 실패 |
| `test_boy_subject_filter_reads_across_break` | 실패 |
| 기존 테스트 38개 | 통과 |
| 합계 | 12 failed, 41 passed |
