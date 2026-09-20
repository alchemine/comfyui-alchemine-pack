# #35 SeparateLoraTags drops the line break before a lora tag

이슈: https://github.com/alchemine/comfyui-alchemine-pack/issues/35

## 이슈
`SeparateLoraTags`가 줄 첫머리에 있는 LoRA 태그를 떼어 낼 때 그 줄의 줄바꿈도 함께 지운다.

| 입력 | 결과 | 기대 |
|---|---|---|
| `1girl,\n<lora:a:0.8>, smile` | `1girl, smile` | `1girl,\nsmile` |
| `1girl,\n<lora:a:0.8> blonde, sky` | `1girl, blonde, sky` | `1girl,\nblonde, sky` |
| `1girl,\n\n<lora:a:0.8>, smile` | `1girl, smile` | `1girl,\n\nsmile` |

원인은 LoRA 태그를 지우는 정규식이 태그 앞의 공백(`\s*`, `[,\s]*`)을 함께 지우는 것이다. 이 공백에 줄바꿈이 들어 있다. #27과 같은 종류의 문제지만, 이 노드는 쉼표 조각이 아니라 정규식으로 동작해서 `join_kept`를 쓸 수 없다.

## 해결책
- 프롬프트를 줄 단위로 나눠서, 기존의 두 정규식을 줄 안에서만 적용한다. 정규식이 줄바꿈을 볼 일이 없어진다.
- LoRA로 시작하던 줄은 그 뒤에 오던 내용으로 시작한다. 앞에 남는 쉼표와 공백은 떼고, 들여쓰기는 지킨다.
- LoRA 태그만 있던 줄은 줄째로 지운다. 빈 줄을 남기지 않는다.
- LoRA가 없는 줄은 건드리지 않는다. 사용자가 넣은 빈 줄도 그대로다.
- 한 줄짜리 프롬프트의 결과는 이전과 같다.

## 테스트 계획
테스트는 `tests/issue/35-separateloratags-drops-the-line-break-before-a-lora-tag/`에 있다.

| 테스트 | 무엇을 테스트하는가 | 기대 결과 |
|---|---|---|
| `test_the_lines_stay_where_they_were` (8개) | LoRA가 줄 첫머리에 있는 경우 3개, LoRA만 있는 줄 2개, LoRA가 줄 중간이나 끝에 있는 경우 3개 | 줄바꿈이 제자리에 남고, LoRA만 있던 줄은 사라진다 |
| `test_one_line_prompts_come_out_as_before` (5개) | 한 줄짜리 프롬프트 4개와 LoRA가 없는 여러 줄 프롬프트 | 이전과 같은 결과 |
| `test_the_example_in_the_docstring` | 클래스 docstring의 예시 | docstring에 적힌 출력 |

실행 방법:

```bash
uv venv
uv pip install --python .venv/bin/python -r requirements.txt -r tests/requirements.txt
.venv/bin/python -m pytest -c tests/pytest.ini tests
```

## 테스트 결과
| | 수정 전 (`346038c`) | 수정 후 |
|---|---|---|
| `test_the_lines_stay_where_they_were` | LoRA가 줄 첫머리에 있는 3개 실패, 나머지 5개 통과 | 8개 모두 통과 |
| `test_one_line_prompts_come_out_as_before` | 통과 | 통과 |
| `test_the_example_in_the_docstring` | 통과 | 통과 |
| 기존 테스트 76개 | 통과 | 통과 |
| 합계 | 3 failed, 87 passed | 90 passed |
