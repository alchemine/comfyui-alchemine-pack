"""
### 이슈
1. `TrainSdxlLora`의 저장 간격과 샘플 간격이 step으로만 정해진다.
   * ASIS: `save_unit=epochs, save_steps=2`를 넣어도 2 step마다 저장한다.
   * TOBE: 2 epoch마다(이미지 5장, batch 1이면 10 step마다) 저장한다.

### 원인 분석
1. `save_steps`, `sample_steps`에 단위 입력이 없어 값이 그대로 step으로 넘어간다.

### 솔루션
1. 선택 입력 `save_unit`, `sample_unit`이 `epochs`이면 `ceil(이미지 수 / batch_size)`를 곱해 step으로 바꾼다.

### 테스트
1. 시나리오: 이미지 폴더를 두고 `TrainSdxlLora`를 실행해, sd-scripts에 넘긴 `training_config.toml`의 간격을 읽는다.
2. 예제 10건: 이미지 수, batch_size, 단위 조합마다 1건이다.
    단위 없이 save 5, sample 3 (옛 워크플로)        <- 둘 다 그대로
    steps 단위로 save 5, sample 3                     <- 둘 다 그대로
    2장, epochs 단위로 save 1, sample 1               <- 2, 2
    5장, save만 epochs 2, sample은 steps 7            <- 10, 7
    5장, save는 steps 7, sample만 epochs 3            <- 7, 15
    5장 batch 2, epochs save 1, sample 2              <- 마지막 batch가 작아 1 epoch = 3 step
    4장 batch 2, epochs save 1, sample 1              <- 2, 2
    1장, epochs save 3, sample 3                      <- 3, 3
    epochs 단위로 save 0, sample 0                    <- 둘 다 끔
    epochs save 1, steps sample 0                     <- save만
3. 비교: `save_every_n_steps`, `sample_every_n_steps`가 기대한 step 수이고, 0이면 키가 없다.
"""

import pytest


@pytest.mark.parametrize(
    "images, over, save_every, sample_every",
    [
        (2, {"save_steps": 5, "sample_steps": 3}, 5, 3),
        (
            2,
            {
                "save_unit": "steps",
                "save_steps": 5,
                "sample_unit": "steps",
                "sample_steps": 3,
            },
            5,
            3,
        ),
        (
            2,
            {
                "save_unit": "epochs",
                "save_steps": 1,
                "sample_unit": "epochs",
                "sample_steps": 1,
            },
            2,
            2,
        ),
        (5, {"save_unit": "epochs", "save_steps": 2, "sample_steps": 7}, 10, 7),
        (5, {"save_steps": 7, "sample_unit": "epochs", "sample_steps": 3}, 7, 15),
        (
            5,
            {
                "batch_size": 2,
                "save_unit": "epochs",
                "save_steps": 1,
                "sample_unit": "epochs",
                "sample_steps": 2,
            },
            3,
            6,
        ),
        (
            4,
            {
                "batch_size": 2,
                "save_unit": "epochs",
                "save_steps": 1,
                "sample_unit": "epochs",
                "sample_steps": 1,
            },
            2,
            2,
        ),
        (
            1,
            {
                "save_unit": "epochs",
                "save_steps": 3,
                "sample_unit": "epochs",
                "sample_steps": 3,
            },
            3,
            3,
        ),
        (
            5,
            {
                "save_unit": "epochs",
                "save_steps": 0,
                "sample_unit": "epochs",
                "sample_steps": 0,
            },
            None,
            None,
        ),
        (5, {"save_unit": "epochs", "save_steps": 1, "sample_steps": 0}, 5, None),
    ],
)
def test_save_and_sample_intervals_count_in_their_own_unit(
    train, images, over, save_every, sample_every
):
    config, _ = train(images, **over)
    assert config.get("save_every_n_steps") == save_every
    assert config.get("sample_every_n_steps") == sample_every
