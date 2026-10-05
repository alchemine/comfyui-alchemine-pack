# ComfyUI-Alchemine-Pack

[ComfyUI](https://github.com/comfyanonymous/ComfyUI)를 위한 커스텀 노드 팩입니다. 프롬프트 처리, LoRA 태그 로딩, 워크플로우 제어, 이미지 보정, 입력 브로드캐스트 등 다양한 유틸리티 노드를 제공합니다.

## 관련 팩

| 팩 | 노드 |
|----|------|
| [comfyui-generator-pack](https://github.com/alchemine/comfyui-generator-pack) | Danbooru 데이터셋으로 태그 제안, 문장을 태그로 변환, 캐릭터 태그 뽑기: Tags Generator, Tags Extractor, Character Tags Generator, Tags Conflict Filter, Classify Tags, Group Tags |
| [comfyui-daam-pack](https://github.com/alchemine/comfyui-daam-pack) | 프롬프트 태그별 cross-attention 히트맵: Sampler Custom (DAAM), DAAM Tag Explorer |
| [comfyui-danbooru-pack](https://github.com/alchemine/comfyui-danbooru-pack) | Danbooru 태그 조회와 포스트 다운로드 |
| [comfyui-evaluate-pack](https://github.com/alchemine/comfyui-evaluate-pack) | Evaluate: 노드에 쓴 Python 코드로 문자열을 변환 |
| [comfyui-api-pack](https://github.com/alchemine/comfyui-api-pack) | 외부 API: zero-shot 분류, OpenAI Inference, 원격 ComfyUI API (Api Generate / Submit / Collect, Load Workflow), Grok 이미지-투-비디오 |

이 팩에서 옮겨 간 노드는 노드 id가 그대로라서, 그 노드가 있는 팩을 설치하면 기존 워크플로가 그대로 열립니다.

## 설치 방법

1. 이 저장소를 ComfyUI의 `custom_nodes` 디렉터리에 클론하거나 복사합니다.
2. ComfyUI를 재시작합니다.

## 제공 노드

### 프롬프트 노드 (`AlcheminePack/Prompt`)

![Prompt Workflow](workflows/comfyui-alchemine-pack-workflow-Prompt.png)

| 노드 | 설명 |
|------|------|
| **ProcessTags** | 태그 처리 전체 파이프라인. ReplaceUnderscores → FilterTags → FilterSubtags → FilterColors → FilterPlurals → SDXLAutoBreak 순서로 처리합니다. |
| **FilterTags** | 블랙리스트 태그를 프롬프트에서 제거합니다. `resources/wildcards.yaml`에 정의된 와일드카드를 지원합니다. |
| **FilterSubtags** | 중복/불필요한 서브태그를 제거합니다 (예: `dog, white dog` → `white dog`). |
| **ReplaceUnderscores** | 모든 언더스코어(`_`)를 공백으로 변환합니다. |
| **FixBreakAfterTIPO** | TIPO 출력 후 BREAK 토큰 형식을 수정합니다 (`(BREAK:-1)` 같은 가중치 제거). |
| **SDXLTokenAnalyzer** | 프롬프트의 CLIP 토큰을 분석합니다 (SDXL 전용). g/l 토크나이저 결과와 토큰 수를 반환합니다. |
| **RemoveWeights** | 모든 가중치 표기를 제거합니다 (예: `(cat:1.2)` → `cat`). |
| **FilterColors** | 같은 대상에 색이 여러 개면 처음 것만 남깁니다 (`red dress, blue dress` → `red dress`). 색 목록은 `resources/wildcards.yaml`의 `color` 키입니다. |
| **FilterPlurals** | 단어 끝의 `s`만 다른 두 태그(`arm up, arms up`)가 있으면 처음 것만 남깁니다. |
| **BoySubjectFilter** | 남성이 필요한 태그(`sex`, `hetero`, `penis`, `another`가 들어간 태그 등)가 있으면 `solo`를 지우고, 남성 인물 태그가 없으면 `((1boy))`와 `add_tags`를 넣습니다. |
| **SDXLAutoBreak** | 각 세그먼트가 75토큰 이내가 되도록 자동으로 BREAK를 삽입합니다 (SDXL 전용). |
| **SubstituteTags** | 정규식 기반 태그 치환. 조건부 실행(`run_if`, `skip_if`) 지원. |
| **SeparateLoraTags** | 프롬프트에서 lora 태그(`<lora:...>`)를 분리합니다. 동일한 lora가 여러 번 등장하면 마지막 가중치를 사용합니다. |
| **TextPrompt** | `dynamicPrompts`를 끈 순수 멀티라인 텍스트 입력. `{a|b}`를 입력해도 커서가 끝으로 튀지 않으며, 와일드카드는 실행 시점에 Python에서 확장됩니다. |

#### ProcessTags

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `text` | STRING | (필수) | 입력 프롬프트 텍스트 |
| `replace_underscores` | BOOLEAN | True | 언더스코어를 공백으로 변환 |
| `filter_tags` | BOOLEAN | True | 블랙리스트 태그 제거 |
| `filter_subtags` | BOOLEAN | True | 중복/불필요 서브태그 제거 |
| `filter_colors` | BOOLEAN | True | 같은 대상의 색 중복 제거 (FilterColors) |
| `filter_plurals` | BOOLEAN | True | 단수형·복수형만 다른 태그 중 나중 것 제거 (FilterPlurals) |
| `auto_break` | BOOLEAN | False | 75토큰 제한을 위한 자동 BREAK 삽입 |
| `clip` | CLIP | (선택) | `auto_break` 사용 시 필요 |
| `blacklist_tags` | STRING | "" | 쉼표로 구분된 블랙리스트 (와일드카드 지원) |
| `fixed_tags` | STRING | "" | 필터링에 관계없이 보존할 태그 |

| 출력 | 설명 |
|------|------|
| `processed_text` | 처리된 프롬프트 텍스트 |
| `filtered_tags_list` | 제거된 태그 묶음의 리스트 (FilterTags / FilterSubtags / FilterColors / FilterPlurals 단계에서 각각) |

> ⚠️ **6.0.0:** `filter_colors`와 `filter_plurals`가 `filter_subtags`와 `auto_break` 사이에 들어갔습니다. 6.0.0 이전에
> 저장한 워크플로에서는 `auto_break`, `blacklist_tags`, `fixed_tags`의 값이 위젯 두 칸만큼 밀리므로 다시 설정해야 합니다.
> API 형식 워크플로에는 새 입력 두 개를 추가해야 합니다.

#### FilterTags

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `text` | STRING | (필수) | 입력 프롬프트 텍스트 |
| `blacklist_tags` | STRING | "" | 쉼표로 구분된 블랙리스트 (와일드카드 지원) |
| `fixed_tags` | STRING | "" | 필터링에 관계없이 보존할 태그 |

| 출력 | 설명 |
|------|------|
| `processed_text` | 블랙리스트 태그가 제거된 프롬프트 |
| `filtered_tags` | 제거된 태그들의 쉼표 구분 목록 |

#### FilterSubtags

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `text` | STRING | (필수) | 입력 프롬프트 텍스트 |
| `fixed_tags` | STRING | "" | 서브태그 필터링에서도 보존할 태그 |

| 출력 | 설명 |
|------|------|
| `processed_text` | 불필요한 서브태그가 제거된 프롬프트 |
| `filtered_tags` | 제거된 서브태그들의 쉼표 구분 목록 |

#### SDXLTokenAnalyzer

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `clip` | CLIP | (필수) | `clip_g` / `clip_l`를 모두 포함하는 CLIP 모델 |
| `text` | STRING | (필수) | 입력 프롬프트 텍스트 |

| 출력 | 설명 |
|------|------|
| `g_tokens` | `clip_g` 토크나이저 결과 (BREAK 기준 세그먼트 분리) |
| `g_token_count` | 세그먼트별 `clip_g` 토큰 수 (쉼표 구분) |
| `l_tokens` | `clip_l` 토크나이저 결과 |
| `l_token_count` | 세그먼트별 `clip_l` 토큰 수 (쉼표 구분) |

#### SDXLAutoBreak

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `clip` | CLIP | (필수) | CLIP 모델 (`clip_g` 토큰 수 기준) |
| `text` | STRING | (필수) | 입력 프롬프트 텍스트 |

#### ReplaceUnderscores / FixBreakAfterTIPO / RemoveWeights

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `text` | STRING | (필수) | 입력 프롬프트 텍스트 |

#### FilterColors / FilterPlurals

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `text` | STRING | (필수) | 입력 프롬프트 텍스트 |

| 출력 | 설명 |
|------|------|
| `processed_text` | 중복된 태그를 지운 프롬프트 |
| `filtered_tags` | 지워진 태그 |

#### BoySubjectFilter

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `text` | STRING | (필수) | 입력 프롬프트 텍스트 |
| `add_tags` | STRING | "(hetero:1.1), (couple:1.1), (deep skin:1.1)" | 남성 인물 태그가 없을 때 `((1boy))` 뒤에 넣는 태그 |

남성이 필요한 태그가 없는 프롬프트는 그대로 돌려줍니다. 태그는 전체가 일치해야 하며(`sex toy`, `sexy`는 해당 없음),
`after `로 시작하는 태그는 건너뛰고, 밑줄과 대소문자는 구분하지 않습니다.

#### SubstituteTags

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `text` | STRING | (필수) | 입력 프롬프트 텍스트 |
| `pattern` | STRING | "" | 매칭할 정규식 패턴 |
| `repl` | STRING | "" | 대체 문자열 |
| `run_if` | STRING | "" | 이 패턴이 있을 때만 실행 |
| `skip_if` | STRING | "" | 이 패턴이 있으면 건너뜀 |

#### SeparateLoraTags

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `text` | STRING | (필수) | 입력 프롬프트 텍스트 |

| 출력 | 설명 |
|------|------|
| `text_without_lora` | lora 태그가 제거된 텍스트 (원본 공백/줄바꿈 최대한 유지) |
| `text_with_lora` | 중복 제거된 lora 태그들을 공백으로 join한 문자열 (동일 lora는 마지막 가중치 사용) |

#### TextPrompt

ComfyUI 기본 텍스트 위젯은 `dynamicPrompts`가 켜져 있어 입력 중에 필드를
재파싱합니다. `{a|b}`를 타이핑할 때마다 커서가 끝으로 이동하는 원인이죠. 이
노드는 그 플래그를 꺼서 위젯을 일반 텍스트 박스처럼 동작하게 하고, 대신
`{option1|option2|...}` 문법을 실행 시점에 Python에서 해석합니다(그룹당 무작위
1개 선택, 중첩 지원).

[ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)이 설치되어
있으면 그쪽 와일드카드 엔진을 대신 사용해 `__wildcard__` 파일 참조, `$$` 다중
선택, `#` 주석까지 해석합니다 — TextPrompt 하나로 ImpactWildcardProcessor를
대체할 수 있습니다. 없으면 `{a|b}` 문법만 해석합니다.

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `text` | STRING | (필수) | 멀티라인 프롬프트. `{a|b}` 그룹은 실행 시 확장 |
| `seed` | INT | 0 | 입력 전용. 와일드카드 선택을 고정해 재현 가능 |

| 출력 | 설명 |
|------|------|
| `text` | 모든 와일드카드 그룹이 해석된 프롬프트 |

---

### 이미지 노드 (`AlcheminePack/Image`)

#### AdjustImage

색 보정, 샤픈/디노이즈 필터 모음, 선택적 리사이즈를 한 노드에 담았습니다. 모든
값의 기본이 무변화라 슬라이더를 움직이기 전까지는 이미지를 그대로 통과시킵니다.
모든 연산이 이미지가 있는 디바이스의 torch 연산이라 GPU 텐서가 numpy를 오가지
않으며, RGB만 처리하고 알파 채널은 그대로 보존합니다.

처리 순서는 의도적으로 고정입니다: brightness → contrast → saturation → gamma →
denoise → edge enhance → CAS → local contrast → resize. 디노이즈가 샤픈보다
먼저라, 제거하려던 노이즈를 샤프너가 증폭하는 일이 없습니다.

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `image` | IMAGE | (필수) | 입력 이미지 |
| `brightness` | FLOAT | 1.0 | 1.0 = 무변화, <1.0 어둡게, >1.0 밝게 |
| `contrast` | FLOAT | 1.0 | 중간 회색 0.5 기준 스케일 |
| `saturation` | FLOAT | 1.0 | 0.0 = 흑백, >1.0 = 더 선명하게 |
| `gamma` | FLOAT | 1.0 | <1.0 중간톤 밝게, >1.0 어둡게 |
| `edge_enhance` | FLOAT | 0.0 | 고정 엣지 강화 커널의 블렌드 비율 |
| `cas` | FLOAT | 0.0 | Contrast Adaptive Sharpening (AMD FidelityFX). 평탄한 영역을 더, 이미 선명한 엣지는 덜 샤픈해 선화가 바스러지지 않음 |
| `local_contrast` | FLOAT | 0.0 | 큰 반경 언샤프("clarity"). 중간 스케일의 깊이감 추가 |
| `denoise` | FLOAT | 0.0 | 엣지 보존 bilateral 필터. 선은 유지하며 평탄 노이즈와 밴딩 제거 |
| `upscale_method` | COMBO | lanczos | `scale_by` != 1.0일 때의 리샘플링 방식 |
| `scale_by` | FLOAT | 1.0 | 배율. 1.0 = 리사이즈 없음 |

| 출력 | 설명 |
|------|------|
| `image` | 보정된 이미지 |

---

### Everywhere 노드 (`AlcheminePack/Everywhere`)

| 노드 | 설명 |
|------|------|
| **Everywhere** | 고정된 이름의 입력(`model`, `clip`, `vae`, `positive`, `negative`, `latent_image`, `seed`, `blacklist`)을 갖는 [cg-use-everywhere](https://github.com/chrisgoringe/cg-use-everywhere) 브로드캐스터. 입력 이름이 고정되어 UE의 이름 기반 라우팅이 유지되므로, 두 컨디셔닝을 수동 이름 변경 없이 구분해 전달합니다. cg-use-everywhere 필요. |

---

### 플로우 컨트롤 노드 (`AlcheminePack/FlowControl`)

![Flow Control Workflow](workflows/comfyui-alchemine-pack-workflow-FlowControl.png)

| 노드 | 설명 |
|------|------|
| **Lazy Execution** | `signal`이 해결된 후에만 `value`를 전달합니다. 실행 순서를 제어하며, 상류 노드가 `signal`로 보낸 `ExecutionBlocker`를 그대로 전파해 하류 노드를 막습니다. |

#### Lazy Execution

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `value` | ANY | 전달할 값 |
| `signal` | ANY | 게이트 입력 — 이 입력이 해결되어야 `value`가 전달됨 |

| 출력 | 설명 |
|------|------|
| `value` | `signal`이 해결된 뒤 그대로 전달되는 `value` 입력 |

**사용 사례:** 순차 실행이 필요할 때(예: 생성 A가 완료된 후에만 생성 B 실행), 또는 상류 노드가 `ExecutionBlocker`를 반환하는 동안 하류 분기를 건너뛰고 싶을 때(예: [comfyui-api-pack](https://github.com/alchemine/comfyui-api-pack)의 **Api Submit**을 **Api Collect**에 물려, 직전 작업이 끝난 뒤에만 새 작업을 제출하도록 게이트).

> **게이트 그래프 점화를 위한 mute:** `value`가 **첫 번째** 입력이므로, 이 노드를 mute(bypass)하면 `signal`을 무시하고 `value`가 그대로 통과합니다. 영원히 막혀 있을 루프를 콜드 스타트할 때 유용합니다 — 예를 들어 **Api Collect**에 아직 작업이 없어 계속 `ExecutionBlocker`를 내보낼 때, 한 번만 게이트를 mute해서 첫 **Api Submit**을 발사하고, 다시 un-mute해 정상 게이트로 복귀합니다.

---

### Lora 노드 (`AlcheminePack/Lora`) *(실험적)*

| 노드 | 설명 |
|------|------|
| **DownloadImage** | URL의 이미지를 ComfyUI output 디렉터리로 다운로드합니다. |
| **SaveImageWithText** | 이미지와 `.txt` 캡션 파일을 함께 저장합니다 (LoRA 학습 데이터셋 준비용). |

#### DownloadImage

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `url` | STRING | (필수) | 다운로드할 이미지 URL |
| `dir_path` | STRING | "output/images" | 저장 디렉터리 (ComfyUI output 기준 상대 경로) |

| 출력 | 설명 |
|------|------|
| `image` | 로드된 이미지 텐서 |
| `file_path` | 저장된 파일 경로 (output 기준 상대 경로) |

#### SaveImageWithText

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `image` | IMAGE | (필수) | 저장할 이미지 |
| `text` | STRING | (필수) | 동일 이름의 `.txt`로 저장될 캡션 |
| `dir_path` | STRING | (필수) | 저장 디렉터리 (ComfyUI output 기준 상대 경로) |
| `prefix` | STRING | "" | 파일명 접두사 (설정 시 자동 인덱스 부여) |

| 출력 | 설명 |
|------|------|
| `image_path` | 저장된 `.png` 경로 |
| `text_path` | 저장된 `.txt` 경로 |

---

### Model 노드 (`AlcheminePack/Model`)

| 노드 | 설명 |
|------|------|
| **Cached Load LoRA Tag** | 텍스트의 `<lora:name:weight>` 태그가 가리키는 LoRA를 로드하고 패치 결과를 캐싱합니다. |

#### Cached Load LoRA Tag

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `model` | MODEL | (필수) | 패치할 베이스 모델 |
| `clip` | CLIP | (필수) | 패치할 베이스 CLIP |
| `text` | STRING (multiline) | (필수) | `<lora:...>` 태그가 포함된 프롬프트 |

| 출력 | 설명 |
|------|------|
| `MODEL` | 해당 LoRA로 패치된 모델 |
| `CLIP` | 해당 LoRA로 패치된 CLIP |
| `STRING` | 모든 lora 태그가 제거된 프롬프트 |

- 태그 형식: `<lora:name:model_weight:clip_weight>` — clip weight는 선택이며 생략 시 model weight를 따름. 숫자 weight가 아예 없는 태그(`<lora:name>`)는 lora 태그로 인식되지 않아 로드되지도, 출력 텍스트에서 제거되지도 않습니다. 가중치 0으로 로드하려면 `<lora:name:0>`을 쓰세요.
- `name`은 `loras` 폴더 파일명의 접두사로 매칭되며, 매칭되지 않는 태그는 건너뜁니다.
- `text`/`model`/`clip`이 그대로면 패치 결과를 캐시에서 반환해 LoRA 재로드·재패치를 생략합니다.

---

## 와일드카드 지원

`FilterTags`와 `ProcessTags` 노드는 `resources/wildcards.yaml`에 정의된 와일드카드를 지원합니다.

**예시:** 블랙리스트에 `<color>`를 사용하면 YAML 파일에 정의된 모든 색상으로 펼쳐져 하나의 정규식으로 합쳐집니다. `(shiny|dark|colored|<color>) skin` 한 줄로 `blue skin`, `grey skin`, `two-tone skin` 등이 모두 차단됩니다.

와일드카드 팩이 쓰는 `__color__`가 아니라 꺾쇠 괄호입니다. 와일드카드 프로세서(ImpactWildcardProcessor, Dynamic Prompts)는 이 노드보다 **앞에서** 실행되므로 FilterTags가 보기 전에 `__color__`를 소비해버리고, 그것도 **하나만 뽑는** 방식이라 블랙리스트가 원하는 것과 정반대입니다. `<...>`는 어떤 와일드카드 프로세서도 사용하지 않는 문법이라 그대로 통과해 이 노드까지 도달합니다. 와일드카드 프로세서를 거치지 않는 블랙리스트라면 `__color__` 형태도 계속 동작합니다.

## 예시

### ProcessTags 예시

```
입력: dog, cat, white dog, black cat
블랙리스트: ^cat$
출력: white dog, black cat
필터됨: ['cat', 'dog']   (단계별 한 칸: FilterTags, 그다음 FilterSubtags)
```

블랙리스트 토큰은 태그 내부 어디든 매칭되는 정규식이라, 그냥 `cat`을 쓰면
`black cat`까지 제거됩니다 — 태그를 정확히 맞추려면 `^cat$`처럼 앵커를 쓰세요.

### FilterSubtags 예시

```
입력: dog, cat, white dog, black cat
출력: white dog, black cat
('dog'와 'cat'이 'white dog'와 'black cat'의 서브태그이므로 제거됨)
```

### SeparateLoraTags 예시

```
입력:
moriaruruka, <lora:characters\lulurka\1-moriaruruka.safetensors:0.7> blonde, gradient hair, jewelry,
<lora:characters\lulurka\2-moriaruruka.safetensors:0.7> <lora:characters\lulurka\3-moriaruruka.safetensors:0.7> <lora:characters\lulurka\3-moriaruruka.safetensors:1.0>

출력:
text_without_lora: moriaruruka, blonde, gradient hair, jewelry
text_with_lora: <lora:characters\lulurka\1-moriaruruka.safetensors:0.7> <lora:characters\lulurka\2-moriaruruka.safetensors:0.7> <lora:characters\lulurka\3-moriaruruka.safetensors:1.0>
```

- lora 블록 뒤에 `,`가 따라오면 앞쪽 콤마/공백까지 함께 제거하여 이중 콤마를 방지합니다.
- lora 블록 뒤에 `,`가 없으면 앞쪽 콤마는 보존하고 선행 공백만 제거합니다.
- 동일한 lora가 여러 번 등장하면 마지막에 지정된 가중치를 사용합니다 (예: 위 예시에서 `3-moriaruruka.safetensors`의 최종 가중치는 `1.0`).

### SubstituteTags 예시

```
# "girl"이 없으면 "1boy"를 "1girl, 1boy"로 교체
pattern: 1boy
repl: 1girl, 1boy
skip_if: girl
```

## 라이선스

GPL-3.0 License
