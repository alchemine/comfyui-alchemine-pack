# Tag Consistency Guard — 데이터 파이프라인

ConsistencyGuard/ClassifyTags 노드(`nodes/prompt.py`, 로직은
`nodes/lib/tag_guard.py`)가 사용하는 데이터 아티팩트의 생성 스크립트 모음.
런타임 코드는 여기 없고 전부 `nodes/lib/`에 있다.

## 아티팩트

| 파일 | 생성 스크립트 |
|---|---|
| `resources/tag_cooc.npz` | `precompute_conflicts.py` |
| `resources/tag_groups.json` | `extract_tag_groups.py` |
| `resources/avoidance_v1.npz` | `precompute_avoidance.py` |
| `resources/subject_joint_v1.npz` | `precompute_subject_joint.py` |

## avoidance_v1.npz — TagSuggest 회피 테이블

- **소스**: `precompute_conflicts.py`가 만든 C 행렬 캐시를 그대로 재사용.
  vocab 정렬 기준이 공백 형태(`computer keyboard`)라는 점에 주의 — 밑줄 형태로
  정렬하면 인덱스가 어긋나고, 형태가 같아 에러 없이 통계만 틀린다.
- **왜 필요한가**: suggest 아티팩트의 척력 테이블은 태그당 512칸 고정이라 행이
  포화된다. `bar_(place)`의 가장 약한 저장 이웃이 lift 0.81이라, 아예 같이 안
  나오는 쌍은 테이블 밖으로 밀려나 런타임에서 안 보인다.
- **판정 규칙**: 포아송 하위꼬리 유의성(p <= alpha) AND lift < 0.35. 효과 크기
  조건이 없으면 빈도 큰 쌍이 20%만 모자라도 유의해져서 `bar`가 `skirt`,
  `navel`을 금지한다 — 참이지만 금지 사유는 아니다.
- alpha는 쌍마다 p를 저장해 런타임에서 조절 (`tag_avoid.DEFAULT_ALPHA`, 기본 0.01).
  빌드는 0.05까지만 담으므로 그보다 느슨하게는 못 간다.
- **한계**: 기대 공기 횟수가 낮은 쌍은 판정 불가. `bar_(place)/computer_keyboard`는
  기대 1.4회에 관측 0회라 p=0.25로 유의하지 않다. 이웃 클러스터를 합산해 표본을
  보충하는 방법도 시도했으나 실패 — keyboard의 이웃(desk, chair, mug, office lady)이
  bar와 오히려 잘 어울려서 합산 lift가 7.7로 나온다.

## subject_joint_v1.npz — subject 조합 veto 테이블

- **왜 필요한가**: 나머지 테이블은 전부 pairwise이고 샘플러는 문맥 태그별 lift를
  로그 합산(naive Bayes = 곱)한다. 인원수만은 이 방식이 원리적으로 틀린다 —
  모순이 **조합에서만** 존재하기 때문이다.

  ```
  lift(threesome | 1girl) = 0.59    거의 중립
  lift(threesome | 1boy)  = 2.62    오히려 끌어당김
  곱                      = 1.56    어떤 veto 도 통과
  lift(threesome | 둘 다) = 0.02    2,636건 기대에 52건
  ```

  `1girl`은 "사람이 하나"가 아니라 "여자가 하나"라서 `1girl, 2boys, threesome`은
  정상 조합이고, 양쪽 pairwise 항은 각자 정직하다. 데이터를 더 모아도 안 고쳐진다.

- **소스**: 2026 메타데이터 덤프(9.23M 포스트). 다른 테이블이 쓰는 solo 전용
  행렬로는 만들 수 없다 — subject 태그 **둘**이 함께 있는 포스트가 필요하다.
- **범위**: 어휘 내 subject 태그 26개, 그 쌍 중 2,000 포스트 이상인 94개.
- **판정 규칙**: Poisson 하단 꼬리(precompute_avoidance와 동일) + lift 상한
  + **surprise = joint_lift / (pairwise lift 곱) < 0.2**. 마지막 항이 핵심이다.
  이게 없으면 "그런 그림에 잘 안 나오는 것"이 전부 걸려서 첫 빌드가
  `shrimp tempura`, `racecar`, `space shuttle`을 3,019개나 차단했다.
  pairwise 가 이미 막는 것(`yuri` vs 1girl+1boy)은 `MIN_PRODUCT`로 제외한다.
- **MIN_EXPECTED=100의 근거**: 짝/홀 post id 로 절반씩 빌드해 재현율을 쟀다.
  5에서는 53%만 재현되고, 100에서 91%가 재현되며 항목이 7,664 → 907로 준다.
  임계값 5~1000 사이에서 **생성 결과는 동일**했으므로(모순 0, 고유 태그 186개
  동일) 이건 동작이 아니라 위생의 문제다.
- **재빌드**: `python precompute_subject_joint.py` — 원시 카운트를 캐시하므로
  임계값 스윕은 몇 초. `--half even/odd` 로 재현성 검증.

  ```
  1boy + solo   : 1girl, breasts, large breasts, skirt, cleavage, panties ...
  1girl + solo  : 1boy, facial hair, muscular male, beard, stubble, shota ...
  1girl + 1boy  : threesome, group sex
  ```

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
