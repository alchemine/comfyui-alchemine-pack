# #59 Count save, sample and logging intervals in epochs or steps

이슈: https://github.com/alchemine/comfyui-alchemine-pack/issues/59

### 이슈
1. `TrainSdxlLora`의 저장, 샘플, loss 로그 간격이 step으로만 정해진다.
   * ASIS: `train_unit=epochs`, `train_length=44`로 학습해도 `save_steps=200`처럼 step 수를 직접 계산해 넣어야 한다.
   * TOBE: `save_unit=epochs`, `save_steps=2`처럼 epoch 단위로도 정할 수 있다. 기본값은 지금처럼 steps.

### 원인 분석
1. `train_unit`은 있지만 `save_steps`, `sample_steps`, `logging_steps`에는 단위 입력이 없어 늘 step으로 넘어간다.

### 솔루션
1. 간격마다 단위를 고를 수 있게 하고, epoch로 고르면 학습 길이와 같은 방법으로 step으로 바꾼다.
   - 선택 입력 `save_unit`, `sample_unit`, `logging_unit`(`epochs`/`steps`, 기본 `steps`)을 더한다. 선택 입력이라 이 입력이 없는 옛 워크플로는 그대로 step으로 센다.
   - `epochs`이면 `ceil(이미지 수 / batch_size)`를 곱한다(`train_length`와 같은 계산).
2. 화면에서는 `train_length`처럼 `[epochs][steps] 값` 한 줄로 그린다.
   - `web/js/train_sdxl_lora.js`의 `mergeRow`로 `save`, `sample`, `log` 행을 만들고, 행마다 이름(`training length`, `save interval` 등)을 왼쪽에 쓴다.
3. `README.md`, `README_ko.md`의 `TrainSdxlLora` 표에 세 단위 입력을 더한다.

### 대표 예제
45장, batch 1, 44 epoch(1980 step) 학습 기준이다.

| 입력 | Before | After |
|---|---|---|
| `save_unit=epochs`, `save_steps=2` | 에러 (`save_unit`을 모르는 인자로 받음) | `save_every_n_steps = 90` |
| `sample_unit=epochs`, `sample_steps=4` | 에러 | `sample_every_n_steps = 180` |
| `logging_unit=epochs`, `logging_steps=1` | 에러 | loss 점 45, 90, …, 1980 |
| 단위 없이 `save_steps=200` (옛 워크플로) | `save_every_n_steps = 200` | `save_every_n_steps = 200` |
| `batch_size=2`, `save_unit=epochs`, `save_steps=2` | 에러 | `save_every_n_steps = 46` (1 epoch = 23 step) |

### 테스트
테스트: [`tests/issue/59-count-save-sample-and-logging-intervals-in-epochs-or-steps/`](../../../tests/issue/59-count-save-sample-and-logging-intervals-in-epochs-or-steps/)

| 검증 내용 | 개선점 | Before | After |
|---|---|---|---|
| 세 단위 입력이 선택 입력이고 기본값이 `steps`다 | 옛 워크플로가 그대로 동작한다 | 실패 | 통과 |
| 저장, 샘플 간격이 각자의 단위로 step이 된다 (10건) | epoch 단위로 저장, 샘플 간격을 정한다 | 1/10 통과 | 10/10 통과 |
| loss 점 간격이 각자의 단위로 step이 된다 (10건) | epoch 단위로 loss 점을 찍는다 | 0/10 통과 | 10/10 통과 |
| 대표 예제 5건 | 실제 학습 설정에서 세 간격이 epoch로 정해진다 | 1/5 통과 | 5/5 통과 |

새 이슈 테스트 26개 중 2개(8%) → 26개(100%) 통과, 통과율 +92%p. 합계 205 passed → 231 passed(기존 205 + 새 26).

실행 방법:

```bash
uv venv
uv pip install --python .venv/bin/python -r tests/requirements.txt
.venv/bin/python -m pytest -c tests/pytest.ini tests
```

### References
- https://github.com/alchemine/comfyui-alchemine-pack/issues/59
