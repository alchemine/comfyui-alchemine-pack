- Fixes #66

### 이슈
1. 한 장씩 나눈 빈 latent와 장마다 다른 seed를 만들려면 노드 5개를 엮어야 합니다.
   * ASIS: `easy seed`, `EmptyLatentImagePresets`, `LatentUpscaleBy`, `RebatchLatents`, `ListCounter //Inspire`를 그룹 노드로 묶어 씁니다.
   * TOBE: `Latent With Seed` 노드 하나가 같은 latent 리스트와 seed 리스트(`seed`, `seed + 1`, ...)를 내보냅니다.

### 원인 분석
1. 이 팩에는 빈 latent를 리스트로 나누고 seed를 붙이는 노드가 없습니다.

### 솔루션
1. 빈 latent를 한 장씩 나눈 리스트와 장마다 다른 seed 리스트를 함께 내보내는 `Latent With Seed` 노드를 추가했습니다.
   - 입력은 `seed`, `dimensions`, `invert`, `batch_size`, `scale_by`이고, 카테고리는 `AlcheminePack/Latent`입니다.
   - 빈 latent는 값이 0이라 배율은 크기 계산(`round(w // 8 * scale_by)`)으로만 반영하고, `upscale_method` 입력은 두지 않습니다.
   - 장마다 `batch_index [i]`를 붙여, 같은 seed의 noise가 `RebatchLatents` 출력과 같게 나옵니다.

### 대표 예제
| 입력 | Before | After |
|---|---|---|
| 1152 x 896, invert, `scale_by` 1.5, batch 3, seed 885647646956553 | 노드 5개 그룹 | latent 3개(1×4×216×168), seed 885647646956553~885647646956555 |
| 기존 노드 체인과 비교 10건(크기, 값, `batch_index`, noise) | 기준 | 10건 모두 같음 |

### 테스트
[test_latent_with_seed.py](https://github.com/alchemine/comfyui-alchemine-pack/blob/99bcf233b0c98893ba8b53bb4c170d00ba999224/tests/issue/66-add-a-latent-with-seed-node/test_latent_with_seed.py)

| 검증 내용 | 개선점 | Before | After |
|---|---|---|---|
| `dimensions`, `invert`, `scale_by`에 따른 latent 크기 | 기존 노드 체인과 같은 크기 | 노드 없음 | 통과 |
| 장마다 한 장, seed와 `batch_index` | 장마다 다른 seed와 같은 noise | 노드 없음 | 통과 |
| `batch_size` 1 | 한 장만 출력 | 노드 없음 | 통과 |

- 전체 258개 중 258개(100%) 통과. 새 테스트는 10개이고, 추가 전에는 노드가 없어 해당 동작을 확인할 수 없었습니다.

### References
- #66
