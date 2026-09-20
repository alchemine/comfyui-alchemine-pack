# #19 Add BoySubjectFilter node

이슈: https://github.com/alchemine/comfyui-alchemine-pack/issues/19

## 이슈
- 남성이 필요한 태그(`sex`, `hetero`, `penis`, `another`가 들어간 태그 등)가 있는데 프롬프트가 `solo`이거나 남성 인물 태그가 없으면, 인원수와 태그가 서로 맞지 않는다.

## 해결책
- `BoySubjectFilter` 노드를 `AlcheminePack/Prompt`에 추가한다. 노드는 `nodes/prompt.py`에, 로직과 목록은 `nodes/lib/tag_boy.py`에 둔다.
- 입력은 `text`, `add_tags`(기본값 `(hetero:1.1), (couple:1.1), (deep skin:1.1)`)이고, 출력은 `processed_text`다.
- 남성이 필요한 태그가 있을 때만 동작한다.
  - `solo`를 지운다.
  - 남성 인물 태그(`1boy`, `2boys`, `multiple boys` 등)가 없으면 `((1boy))`와 `add_tags`를 끝에 넣는다.
- 탐지 기준
  - `BOY_TAGS`나 `MALE_BODY`에 있는 태그, `BOY_PATTERNS`에 맞는 태그, `another`가 들어간 태그. 태그 전체와 맞춰 보므로 `sex toy`, `sexy`, `unisex`는 걸리지 않는다.
  - `after `로 시작하는 태그는 혼자서도 성립하므로 탐지하지 않는다.
  - 밑줄과 대소문자는 구분하지 않는다.
- 버전을 6.0.0으로 올린다.

## 테스트 계획
테스트는 `tests/issue/19-add-boysubjectfilter-node/`에 있다. 노드의 `execute`를 직접 호출한다.

| 테스트 | 무엇을 테스트하는가 | 기대 결과 |
|---|---|---|
| `test_no_boy_counts_him_in_and_takes_solo_out` | `1girl, solo, sex, smile` | `1girl, sex, smile, ((1boy)), (hetero:1.1), (couple:1.1), (deep skin:1.1)` |
| `test_a_counted_boy_only_loses_solo` | `1girl, 1boy, solo, sex` | `1girl, 1boy, sex` |
| `test_add_tags_is_what_comes_in_with_him` | `add_tags="hetero"` | `((1boy))` 뒤에 `hetero`만 붙는다 |
| `test_left_alone` (4개) | 탐지할 태그 없음, `1girl, 1boy, solo`, `sex toy`·`sexy`·`unisex`, `after sex` | 입력 그대로 |
| `test_detected` (5개) | `another` 태그, 패턴 태그, `erection`, `Sex_From_Behind` | `solo`가 빠지고 남성이 들어간다 |
| `test_a_male_body_counts_even_next_to_futanari` | `1girl, solo, futanari, erection` | `solo`가 빠지고 남성이 들어간다 |
| `test_multiple_boys_is_a_counted_boy` | `1girl, multiple boys, solo, gangbang` | `solo`만 빠진다 |

실행 방법:

```bash
uv venv
uv pip install --python .venv/bin/python -r requirements.txt -r tests/requirements.txt
.venv/bin/python -m pytest -c tests/pytest.ini tests
```

## 테스트 결과
| | 수정 전 (`2de44ed`) | 수정 후 |
|---|---|---|
| 이 이슈의 테스트 14개 | 실패 (`BoySubjectFilter`가 없다) | 통과 |
| #13, #16의 테스트 9개 | 통과 | 통과 |
| 합계 | 14 failed, 9 passed | 23 passed |

`nodes/prompt.py`가 `yaml`을 import하므로 `tests/requirements.txt`에 `pyyaml`을 추가했다.
