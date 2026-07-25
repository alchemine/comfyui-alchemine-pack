# Tag Consistency Guard — 데이터 파이프라인

ConsistencyGuard/ClassifyTags 노드(`nodes/prompt.py`, 로직은 `nodes/lib/tag_guard.py`,
`nodes/lib/tag_classify.py`)가 사용하는 데이터 아티팩트의 생성 스크립트 모음.
런타임 코드는 여기 없고 전부 `nodes/lib/`에 있다.

## 아티팩트

| 파일 | 생성 스크립트 |
|---|---|
| `nodes/resources/tag_cooc.npz` | `precompute_conflicts.py` |
| `nodes/resources/tag_groups.json` | `extract_tag_groups.py` |

## tag_cooc.npz — co-occurrence conflict 테이블

- **소스**: isek-ai/danbooru-tags-2024 parquet 덤프 (`~/workspace/danbooru_dumps/db*.parquet`, ~2GB).
  solo 태그가 붙은 포스트만 집계 — 다인 포스트는 캐릭터별 속성이 섞여
  co-occurrence를 오염시킴 (blonde hair + red hair가 2girls 포스트에선 정상).
- **vocab**: `danbooru_general_tags.csv`에서 post_count >= 100인 general 태그 (~21k).
- **판정 규칙** (런타임, `tag_guard.is_conflict`): PPMI-profile cosine >= 0.75 (같은 축의 태그)
  AND lift < 0.2 (서로 회피). PPMI 컨텍스트 차원은 >=1000 태그 8.5k개로 제한 —
  rare 차원을 섞으면 cos 스케일이 전체적으로 내려가 임계값 캘리브레이션이 깨진다.
  기대 co-occurrence가 E_MIN(15) 미만인 쌍은 회피를 입증할 표본이 없으므로
  lift를 중립(1.0)으로 처리해 conflict로 잡히지 않는다.
- **재빌드**: `python precompute_conflicts.py` (numpy/scipy/pyarrow 필요).
  C 행렬 캐시(`danbooru_dumps/cooc_C_solo_min100.npy`, int32)가 있으면 수 분,
  없으면 전체 덤프 재집계.

## tag_groups.json — wiki 태그 그룹 매핑

- **소스**: `~/workspace/danbooru_dumps/wiki/` 아래 두 HF wiki 덤프
  (deepghs/danbooru_wikis_full의 tag_group:* 페이지 + isek-ai/danbooru-wiki-2024의 역링크).
- ClassifyTags의 9개 버킷 분류에 사용. 그룹은 cooc conflict 규칙을 hard-gate하지 않는다
  (pussy vs pantyshot처럼 cross-group 제거도 정당할 수 있음).
- **재빌드**: `python extract_tag_groups.py`.

## 기타 파일

- `calibration_pairs.txt` — 수동 판정한 conflict/compatible 쌍. 임계값 튜닝 기준.
  알려진 한계: `wading/sitting`(conflict) vs `spread arms/sitting`(compatible)은
  co-occurrence 통계만으로 분리 불가 (open problem).
- `test_tag_guard.py` — `nodes/lib/tag_guard.py` 단위 테스트. `python test_tag_guard.py`.
- `verify_danbooru_tags.py` — tag_data.py 정적 목록을 Danbooru wiki와 대조하는 검증
  스크립트 (playwright 필요; 이 서버에서 Danbooru 직접 접근은 IP 차단 — HF 덤프 사용).
- `danbooru_general_tags.csv` — general 태그 이름/post_count 스냅샷 (vocab 소스).
