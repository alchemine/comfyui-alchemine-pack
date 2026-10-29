"""
### 이슈
1. `TrainSdxlLora`의 loss 차트 간격이 step으로만 정해진다.
   * ASIS: `logging_unit=epochs, logging_steps=1`을 넣어도 매 step 점을 찍는다.
   * TOBE: 1 epoch마다(이미지 3장, batch 1이면 3 step마다) 점을 찍고, 마지막 step은 항상 찍는다.

### 원인 분석
1. `logging_steps`에 단위 입력이 없어 값이 그대로 step으로 쓰인다.

### 솔루션
1. 선택 입력 `logging_unit`이 `epochs`이면 `ceil(이미지 수 / batch_size)`를 곱해 step으로 바꾼다.

### 테스트
1. 시나리오: 10 step을 학습하며 노드가 보내는 `alchemine.train_progress` 이벤트의 step을 모은다.
2. 예제 10건: 이미지 수, batch_size, 간격, 단위 조합마다 1건이다.
    3장, steps 1                 <- 매 step
    3장, steps 4                 <- 4, 8, 10
    3장, epochs 1                <- 3, 6, 9, 10
    2장, epochs 2                <- 4, 8, 10
    5장 batch 2, epochs 1        <- 1 epoch = 3 step
    1장, epochs 1                <- 매 step
    4장 batch 2, epochs 1        <- 2 step마다
    10장, epochs 1               <- 마지막 step만
    7장 batch 3, epochs 2        <- 6, 10
    12장, epochs 1               <- 1 epoch이 전체보다 길어 마지막 step만
3. 비교: 보낸 step 목록이 기대한 목록과 같다.
"""

import pytest


@pytest.mark.parametrize(
    "images, over, steps",
    [
        (3, {"logging_steps": 1, "logging_unit": "steps"}, list(range(1, 11))),
        (3, {"logging_steps": 4, "logging_unit": "steps"}, [4, 8, 10]),
        (3, {"logging_steps": 1, "logging_unit": "epochs"}, [3, 6, 9, 10]),
        (2, {"logging_steps": 2, "logging_unit": "epochs"}, [4, 8, 10]),
        (
            5,
            {"logging_steps": 1, "logging_unit": "epochs", "batch_size": 2},
            [3, 6, 9, 10],
        ),
        (1, {"logging_steps": 1, "logging_unit": "epochs"}, list(range(1, 11))),
        (
            4,
            {"logging_steps": 1, "logging_unit": "epochs", "batch_size": 2},
            [2, 4, 6, 8, 10],
        ),
        (10, {"logging_steps": 1, "logging_unit": "epochs"}, [10]),
        (
            7,
            {"logging_steps": 2, "logging_unit": "epochs", "batch_size": 3},
            [6, 10],
        ),
        (12, {"logging_steps": 1, "logging_unit": "epochs"}, [10]),
    ],
)
def test_loss_points_are_counted_in_their_own_unit(train, images, over, steps):
    _, sent = train(images, **over)
    assert sent == steps
