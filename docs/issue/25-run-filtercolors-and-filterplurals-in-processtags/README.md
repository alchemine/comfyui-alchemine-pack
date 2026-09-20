# #25 Run FilterColors and FilterPlurals in ProcessTags

이슈: https://github.com/alchemine/comfyui-alchemine-pack/issues/25

## 이슈
- `ProcessTags`는 `ReplaceUnderscores → FilterTags → FilterSubtags → AutoBreak`를 한 노드에서 실행한다.
- 새로 추가된 `FilterColors`(#21)와 `FilterPlurals`(#23)는 여기에 들어 있지 않아서, 따로 노드를 이어 붙여야 한다.

## 해결책
- `FilterSubtags` 다음에 `FilterColors`, `FilterPlurals`를 차례로 실행한다.
- `filter_subtags` 밑, `auto_break` 위에 `filter_colors`, `filter_plurals` 위젯(BOOLEAN, 기본값 `True`)을 추가한다.
- 지워진 태그는 기존 단계처럼 `filtered_tags_list`에 단계마다 하나씩 이어 붙인다.
- 호환성: 위젯이 중간에 들어가므로 6.0.0 이전에 저장한 워크플로에서 `auto_break`, `blacklist_tags`, `fixed_tags`의 값이 두 칸 밀린다. API 형식 워크플로에는 새 입력 두 개를 추가해야 한다. README에 적는다.

## 테스트 계획
테스트는 `tests/issue/25-run-filtercolors-and-filterplurals-in-processtags/`에 있다. 입력은 `1girl, red_dress, blue dress, arm up, arms up, dog, white dog`다.

| 테스트 | 무엇을 테스트하는가 | 기대 결과 |
|---|---|---|
| `test_both_steps_run_by_default_after_the_subtags` | 기본값으로 실행 | `1girl, red dress, arm up, white dog`, 지워진 묶음은 순서대로 `dog`, `blue dress`, `arms up` |
| `test_each_step_has_its_own_switch` | `filter_colors=False`, `filter_plurals=False` | 끈 단계의 태그만 남는다 |
| `test_the_widgets_sit_under_filter_subtags` | `INPUT_TYPES`의 `required` 순서 | `filter_subtags`, `filter_colors`, `filter_plurals`, `auto_break` |
| `test_break_survives_the_new_steps` | `red dress, smile BREAK blue dress, sky` | `red dress, smile BREAK sky` |

실행 방법:

```bash
uv venv
uv pip install --python .venv/bin/python -r requirements.txt -r tests/requirements.txt
.venv/bin/python -m pytest -c tests/pytest.ini tests
```

## 테스트 결과
| | 수정 전 (`e2b8aaf`) | 수정 후 |
|---|---|---|
| 이 이슈의 테스트 4개 | 실패 | 통과 |
| 기존 테스트 34개 | 통과 | 통과 |
| 합계 | 4 failed, 34 passed | 38 passed |
