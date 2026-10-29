"""
### 이슈
1. epoch로 학습 길이를 정해도 저장, 샘플, loss 로그 간격은 step 수를 직접 계산해 넣어야 한다.

### 원인 분석
1. `save_steps`, `sample_steps`, `logging_steps`에 단위 입력이 없다.

### 솔루션
1. 간격마다 `epochs`/`steps` 단위 입력을 더하고, `train_length`와 같은 방법으로 step으로 바꾼다.

### 테스트
1. 시나리오: 45장, batch 1, 44 epoch(1980 step)로 학습한다.
2. 예제 5건:
    2 epoch마다 저장           <- 직접 찾음 (seamless LoRA 학습 설정, 90 step)
    4 epoch마다 샘플           <- 직접 찾음 (180 step)
    1 epoch마다 loss 점        <- 직접 찾음 (45, 90, ..., 1980)
    단위 없이 save 200         <- 직접 찾음 (이 입력이 생기기 전에 저장한 워크플로)
    batch 2로 2 epoch마다 저장 <- 직접 찾음 (1 epoch = 23 step, 46 step)
3. 비교: 설정 파일의 간격과 보낸 loss 점의 step이 기대한 값이다.
"""

EPOCH = {"train_unit": "epochs", "train_length": 44}


def test_saving_every_two_epochs(train):
    config, _ = train(45, save_unit="epochs", save_steps=2, **EPOCH)
    assert config["max_train_steps"] == 1980
    assert config["save_every_n_steps"] == 90


def test_sampling_every_four_epochs(train):
    config, _ = train(45, sample_unit="epochs", sample_steps=4, **EPOCH)
    assert config["sample_every_n_steps"] == 180


def test_a_loss_point_every_epoch(train):
    _, sent = train(45, logging_unit="epochs", logging_steps=1, **EPOCH)
    assert sent == list(range(45, 1981, 45))


def test_a_workflow_saved_before_the_unit_inputs_keeps_steps(train):
    config, _ = train(45, save_steps=200, **EPOCH)
    assert config["save_every_n_steps"] == 200


def test_a_partial_last_batch_counts_as_a_step(train):
    config, _ = train(45, batch_size=2, save_unit="epochs", save_steps=2, **EPOCH)
    assert config["max_train_steps"] == 44 * 23
    assert config["save_every_n_steps"] == 46
