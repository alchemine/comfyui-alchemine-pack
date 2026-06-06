# ComfyUI-Alchemine-Pack

[ComfyUI](https://github.com/comfyanonymous/ComfyUI)를 위한 커스텀 노드 팩입니다. 프롬프트 처리, Danbooru 연동, LLM 추론, LoRA 태그 로딩, Grok 이미지-투-비디오, 원격 ComfyUI API 실행, 워크플로우 제어 등 다양한 유틸리티 노드를 제공합니다.

## 설치 방법

1. 이 저장소를 ComfyUI의 `custom_nodes` 디렉터리에 클론하거나 복사합니다.
2. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```
3. ComfyUI를 재시작합니다.

## 제공 노드

### 프롬프트 노드 (`AlcheminePack/Prompt`)

![Prompt Workflow](workflows/comfyui-alchemine-pack-workflow-Prompt.png)

| 노드 | 설명 |
|------|------|
| **ProcessTags** | 태그 처리 전체 파이프라인. ReplaceUnderscores → FilterTags → FilterSubtags → SDXLAutoBreak 순서로 처리합니다. |
| **FilterTags** | 블랙리스트 태그를 프롬프트에서 제거합니다. `resources/wildcards.yaml`에 정의된 와일드카드를 지원합니다. |
| **FilterSubtags** | 중복/불필요한 서브태그를 제거합니다 (예: `dog, white dog` → `white dog`). |
| **ReplaceUnderscores** | 모든 언더스코어(`_`)를 공백으로 변환합니다. |
| **FixBreakAfterTIPO** | TIPO 출력 후 BREAK 토큰 형식을 수정합니다 (`(BREAK:-1)` 같은 가중치 제거). |
| **SDXLTokenAnalyzer** | 프롬프트의 CLIP 토큰을 분석합니다 (SDXL 전용). g/l 토크나이저 결과와 토큰 수를 반환합니다. |
| **RemoveWeights** | 모든 가중치 표기를 제거합니다 (예: `(cat:1.2)` → `cat`). |
| **SDXLAutoBreak** | 각 세그먼트가 75토큰 이내가 되도록 자동으로 BREAK를 삽입합니다 (SDXL 전용). |
| **SubstituteTags** | 정규식 기반 태그 치환. 조건부 실행(`run_if`, `skip_if`) 지원. |
| **SeparateLoraTags** | 프롬프트에서 lora 태그(`<lora:...>`)를 분리합니다. 동일한 lora가 여러 번 등장하면 마지막 가중치를 사용합니다. |

#### ProcessTags

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `text` | STRING | (필수) | 입력 프롬프트 텍스트 |
| `replace_underscores` | BOOLEAN | True | 언더스코어를 공백으로 변환 |
| `filter_tags` | BOOLEAN | True | 블랙리스트 태그 제거 |
| `filter_subtags` | BOOLEAN | True | 중복/불필요 서브태그 제거 |
| `auto_break` | BOOLEAN | False | 75토큰 제한을 위한 자동 BREAK 삽입 |
| `clip` | CLIP | (선택) | `auto_break` 사용 시 필요 |
| `blacklist_tags` | STRING | "" | 쉼표로 구분된 블랙리스트 (와일드카드 지원) |
| `fixed_tags` | STRING | "" | 필터링에 관계없이 보존할 태그 |

| 출력 | 설명 |
|------|------|
| `processed_text` | 처리된 프롬프트 텍스트 |
| `filtered_tags_list` | 제거된 태그 묶음의 리스트 (FilterTags / FilterSubtags 단계에서 각각) |

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

---

### Danbooru 노드 (`AlcheminePack/Danbooru`)

> ℹ️ 이 노드들은 순수 `requests`(`danbooru_requests.py`)를 사용합니다 — 브라우저 의존성 없음. Playwright 기반 변형(`danbooru.py`)도 대체용으로 소스에 남겨두었으며, 그걸 쓰려면 `__init__.py`의 import를 바꾸고 `pip install playwright`를 실행하세요. 선택적으로 `.env`의 `WEBSHARE_PROXY_USERNAME` / `WEBSHARE_PROXY_PASSWORD`로 Webshare 프록시를 설정할 수 있습니다.


![Danbooru Workflow](workflows/comfyui-alchemine-pack-workflow-Danbooru.png)

| 노드 | 설명 |
|------|------|
| **Danbooru Post Tags Retriever** | 포스트 ID로 특정 Danbooru 포스트의 태그를 가져옵니다. |
| **Danbooru Related Tags Retriever** | Danbooru에서 빈도/유사도 기반으로 관련 태그를 검색합니다. |
| **Danbooru Popular Posts Tags Retriever** | 인기 포스트(일간/주간/월간)에서 태그를 가져옵니다. |
| **Danbooru Posts Downloader** | 검색 태그 기반으로 Danbooru 이미지를 다운로드합니다. |

> ⚠️ **주의:** 요청을 줄이기 위해 응답을 캐싱합니다 — 특정 post(id 기준)는 프로세스 생존 동안, 가변 엔드포인트(popular / related / search)는 1시간 TTL로 캐싱됩니다. 과도하게 쓰면 여전히 Danbooru 레이트리밋에 걸릴 수 있습니다.

#### Danbooru Post Tags Retriever

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `post_id` | STRING | Danbooru 포스트 ID |

| 출력 | 설명 |
|------|------|
| `full_tags` | 전체 태그 (캐릭터 + 저작권 + 아티스트 + 일반, 메타 제외) |
| `general_tags` | 일반 태그만 |
| `character_tags` | 캐릭터 태그만 |
| `copyright_tags` | 저작권 태그만 |
| `artist_tags` | 아티스트 태그만 |
| `meta_tags` | 메타 태그만 |
| `image_url` | 이미지 URL |

#### Danbooru Related Tags Retriever

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `text` | STRING | (필수) | 입력 태그 |
| `category` | ENUM | "General" | 태그 카테고리 필터 (General/Character/Copyright/Artist/Meta) |
| `order` | ENUM | "Frequency" | 정렬 순서 (Cosine/Jaccard/Overlap/Frequency) |
| `threshold` | FLOAT | 0.3 | 최소 유사도 임계값 |
| `n_min_tags` | INT | 0 | 반환할 최소 태그 수 |
| `n_max_tags` | INT | 100 | 반환할 최대 태그 수 |

#### Danbooru Popular Posts Tags Retriever

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `date` | STRING | "" | 날짜 (YYYY-MM-DD 형식, 비워두면 최신) |
| `scale` | ENUM | "day" | 시간 범위 (day/week/month) |
| `n` | INT | 1 | 가져올 포스트 수 |
| `random` | BOOLEAN | True | `True`: 무작위 `n`개 / `False`: 인기 순위의 `[offset, offset+n)` 구간 |
| `seed` | INT | 0 | 랜덤 시드 (`random=True`일 때만 사용) |
| `offset` | INT | 0 | 인기 순위의 시작 위치 (`random=False`일 때만 사용). `control_after_generate`가 붙어 있어 *increment*로 두면 큐마다 순위를 한 칸씩 내려감. 해당 순위가 없으면 에러. |

출력은 **리스트**(포스트당 한 칸): `full_tags` / `general_tags` / `character_tags` / `copyright_tags` / `artist_tags` / `meta_tags`.

> **팁 — 순위를 하나씩 훑기:** `random=False`, `n=1`, `offset` 컨트롤을 *increment*로 설정. 큐를 누를 때마다 다음 인기글을 반환하며, 그 글이 있는 페이지 1개만 가져옴.

#### Danbooru Posts Downloader

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `tags` | STRING | "" | 검색 태그 |
| `n` | INT | 1 | 다운로드할 이미지 수 |
| `dir_path` | STRING | "" | 출력 디렉터리 (ComfyUI output 폴더 기준 상대 경로) |
| `prefix` | STRING | "" | 파일명 접두사 |

---

### 추론 노드 (`AlcheminePack/Inference`)

![Inference Workflow](workflows/comfyui-alchemine-pack-workflow-Inference.png)

| 노드 | 설명 |
|------|------|
| **OpenAI Inference** | OpenAI 호환 API로 텍스트 생성. 비전 및 씽킹 모드 지원. |

#### OpenAI Inference

OpenAI 호환 백엔드를 하나의 노드로 모두 처리합니다 — OpenAI, vLLM, Ollama의 `/v1` 엔드포인트, Gemini의 OpenAI 호환 엔드포인트. `base_url`/`api_key`/`model`만 원하는 서버로 지정하면 됩니다.

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `prompt` | STRING | "Hello, world!" | 사용자 프롬프트 |
| `system_instruction` | STRING | "You are a helpful assistant." | 시스템 프롬프트 |
| `base_url` | STRING | "" | API base URL, 예: `https://api.openai.com/v1` (`config.json`에 설정 가능) |
| `api_key` | STRING | "" | API 키 (`config.json`에 설정 가능) |
| `model` | STRING | "" | 모델명. 비우면 `/models`에 모델이 하나일 때 자동 감지 |
| `max_output_tokens` | INT | 100 | 최대 출력 토큰 (최대 131072) |
| `seed` | INT | 0 | 랜덤 시드 |
| `temperature` | FLOAT | 0.7 | 샘플링 온도 (0.0–2.0) |
| `think` | BOOLEAN | False | 씽킹 모드 활성화 |
| `image` | IMAGE | (선택) | 비전 작업용 입력 이미지 |

| 출력 | 설명 |
|------|------|
| `response` | 모델의 답변 (`<think>` 블록은 제거됨) |
| `reasoning` | 사고 과정. `reasoning_content` 필드 또는 인라인 `<think>...</think>` 블록에서 추출 (없으면 빈 문자열) |

---

### Evaluate 노드 (`AlcheminePack/Evaluate`)

| 노드 | 설명 |
|------|------|
| **Evaluate** | 사용자 정의 Python 코드를 입력 문자열에 적용해 변환된 결과를 반환합니다. 워크플로우 내 즉석 태그 가공에 유용합니다. |

#### Evaluate

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `tag` | STRING | (필수) | `main(tag)`에 전달될 입력 문자열 |
| `code` | STRING (multiline) | 태그 정렬 스니펫 | `def main(tag: str) -> str`을 정의해야 하는 Python 코드 |

| 출력 | 설명 |
|------|------|
| `tag` | `main(tag)`의 반환값 |

기본 코드는 쉼표로 구분된 태그를 알파벳순으로 정렬합니다:

```python
def main(tag: str) -> str:
    tags = [t.strip() for t in tag.split(",") if t.strip()]
    return ", ".join(sorted(tags))
```

> ⚠️ **보안 주의:** `Evaluate`는 `exec()`로 임의의 Python 코드를 실행합니다. 신뢰할 수 있는 코드만 사용하세요.

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

**사용 사례:** 순차 실행이 필요할 때(예: 생성 A가 완료된 후에만 생성 B 실행), 또는 상류 노드가 `ExecutionBlocker`를 반환하는 동안 하류 분기를 건너뛰고 싶을 때(예: **Api Collect**에 **OpenAI Inference** + **Api Submit** 체인을 물려, 직전 작업이 끝난 뒤에만 새 작업을 만들도록 게이트).

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

### Grok 노드 (`AlcheminePack/Grok`)

| 노드 | 설명 |
|------|------|
| **Grok Generate** | 이미지 한 장으로 Grok Imagine I2V 영상 클립을 만들어 output 폴더에 네이티브 VIDEO로 저장합니다 (노드에서 인라인 미리보기 제공). |
| **Grok Submit** | Fire-and-forget 제출. 생성 요청만 보내고 즉시 `request_id`를 반환합니다 (대기하지 않음). |
| **Grok Collect** | Grok Submit으로 제출한 진행 중 잡을 수집합니다. 준비되면 VIDEO를 반환하고, 아니면 다운스트림을 차단합니다. |

#### Grok Generate

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `image` | IMAGE | (필수) | 소스 이미지 (첫 프레임) |
| `prompt` | STRING | "" | 움직임/연출 설명 (선택) |
| `duration` | INT | 5 | 영상 길이(초) (1–15) |
| `resolution` | ENUM | "720p" | "720p" 또는 "480p" |
| `model` | STRING | "grok-imagine-video-1.5-preview" | Grok 영상 모델 |
| `filename_prefix` | STRING | "grok/GrokVideo" | ComfyUI output 디렉터리 기준 저장 경로 접두사 |
| `poll_interval` | INT | 5 | 상태 폴링 간격(초) (1–60) |
| `timeout` | INT | 600 | 생성 대기 최대 시간(초) (30–3600) |
| `access_token` | STRING | "" | 선택. 비우면 `GROK_ACCESS_TOKEN` 환경변수 사용 |
| `refresh_token` | STRING | "" | 선택. 비우면 `GROK_REFRESH_TOKEN` 환경변수 사용 |
| `client_id` | STRING | "" | 선택. 비우면 `GROK_CLIENT_ID` 환경변수 사용 |

| 출력 | 설명 |
|------|------|
| `video` | 생성된 클립(소리 포함). 노드에서 인라인 미리보기로도 표시됨 |

> **자격증명:** 세 토큰을 노드 입력으로 직접 넣거나, 비워 두면 `GROK_ACCESS_TOKEN` / `GROK_REFRESH_TOKEN` / `GROK_CLIENT_ID` 환경변수에서 읽습니다. 401 발생 시 access token은 자동 갱신됩니다.

#### Grok Submit

입력은 **Grok Generate**와 동일하며(`poll_interval`/`timeout` 제외), 선택 `label`이 추가됩니다. OUTPUT_NODE라서 출력을 소비하는 노드가 없어도 실행됩니다. mp4 저장 경로는 제출 시점에 예약되어 lock에 기록되고, 클립이 준비되면 Collect가 그 경로에 저장합니다.

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `label` | STRING | "" | 선택. 잡과 함께 기록되는 라벨 (이후 Grok Collect가 반환) |

| 출력 | 설명 |
|------|------|
| `request_id` | 제출된 잡의 request id (이미 진행 중인 Grok 잡이 있으면 빈 문자열) |

#### Grok Collect

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `wait_sec` | INT | 0 | `0`이면 준비됐을 때만 수집하고 아니면 즉시 건너뜀. 그 외엔 이 시간(초)까지 대기 (0–3600) |
| `poll_interval` | FLOAT | 5.0 | 대기 중 폴링 간격(초) (0.5–60.0) |
| `access_token` | STRING | "" | 선택. 비우면 `GROK_ACCESS_TOKEN` 환경변수 사용 |
| `refresh_token` | STRING | "" | 선택. 비우면 `GROK_REFRESH_TOKEN` 환경변수 사용 |
| `client_id` | STRING | "" | 선택. 비우면 `GROK_CLIENT_ID` 환경변수 사용 |

| 출력 | 설명 |
|------|------|
| `video` | 준비된 클립. 없으면 다운스트림을 건너뛰는 `ExecutionBlocker` |
| `label` | 제출 시 기록된 라벨 |

> **수거 시 자격증명:** Collect도 Grok API를 호출(폴링/다운로드)하므로 토큰을 입력 또는 환경변수에서 다시 읽습니다 — 토큰은 lock 파일에 저장하지 **않습니다**. Grok Generate와 같은 방식으로 넣어 주세요.

> **종류별 단일 진행 잡:** Grok과 [API](#api-노드-alcheminepackapi) 노드는 패키지 디렉터리의 단일 `jobs.lock`을 공유하지만 종류별로 슬롯이 분리됩니다 — Grok 잡과 API 잡이 동시에 진행될 수 있고, 각 종류는 하나만 허용됩니다. 해당 종류의 잡이 이미 진행 중이면 Submit은 건너뛰고, Collect가 완료되면 슬롯을 비웁니다. `/loop` 등으로 Collect를 반복 실행하면 완료된 클립을 받아올 수 있습니다.

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

- 태그 형식: `<lora:name:model_weight:clip_weight>` — clip weight는 선택이며 생략 시 model weight를 따름. weight 없는 태그는 0으로 로드됩니다.
- `name`은 `loras` 폴더 파일명의 접두사로 매칭되며, 매칭되지 않는 태그는 건너뜁니다.
- `text`/`model`/`clip`이 그대로면 패치 결과를 캐시에서 반환해 LoRA 재로드·재패치를 생략합니다.

---

### API 노드 (`AlcheminePack/API`)

워크플로우를 원격 ComfyUI 인스턴스의 HTTP API로 실행합니다 (예: [RunPod](https://www.runpod.io/) 파드 또는 접근 가능한 임의의 ComfyUI). 모든 노드는 UI 워크플로우 포맷이 아니라 **API 포맷** 워크플로우 JSON(ComfyUI 메뉴: "Save (API Format)")을 받습니다. `api_url`은 원격 베이스 URL로, 예: `https://xxxx-8188.proxy.runpod.net/` 또는 `http://127.0.0.1:8188` 입니다.

| 노드 | 설명 |
|------|------|
| **Load Workflow** | `ComfyUI/user/default/workflows/`의 API 포맷 워크플로우 JSON을 읽어 STRING으로 반환합니다. |
| **Api Generate** | 워크플로우를 원격 ComfyUI에 보내 완료될 때까지 기다린 뒤 출력 이미지/프레임을 반환합니다. |
| **Api Submit** | Fire-and-forget 제출. 잡을 기록하고 즉시 `job_id`를 반환합니다 (대기하지 않음). |
| **Api Collect** | Api Submit으로 제출한 진행 중 잡을 수집합니다. 준비되면 프레임을 반환하고, 아니면 다운스트림을 차단합니다. |

#### Load Workflow

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `filename` | ENUM | (필수) | `user/default/workflows/` 아래의 `.json` 파일 |

| 출력 | 설명 |
|------|------|
| `text` | 워크플로우 JSON 내용 |

#### Api Generate

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `workflow` | STRING (입력) | (필수) | API 포맷 워크플로우 JSON 문자열 또는 파일 경로 |
| `positive_prompt` | STRING (입력) | (필수) | 포지티브 프롬프트. `positive_prompt_id`에 주입됨 |
| `positive_prompt_id` | STRING | (필수) | 포지티브 프롬프트를 받을 노드 id |
| `negative_prompt_id` | STRING | "" | `negative_prompt`(제공 시)를 받을 노드 id |
| `output_id` | STRING | "" | `images`/`gifs` 출력을 가져올 노드 id |
| `seed` | INT | -1 | `-1`이면 워크플로우의 기존 시드 유지 |
| `seed_id` | STRING | "" | `seed`/`noise_seed` 입력에 시드를 받을 노드 id |
| `api_url` | STRING | "" | 원격 ComfyUI 베이스 URL. 예: `https://xxxx-8188.proxy.runpod.net/` 또는 `http://127.0.0.1:8188` |
| `image_node_id` | STRING | "" | 업로드한 `image`를 받을 LoadImage 노드 id |
| `timeout_sec` | INT | 300 | 최대 폴링 시간(초) (1–36000) |
| `negative_prompt` | STRING (입력) | "" | 선택. 비어 있으면 건너뜀 |
| `image` | IMAGE | (선택) | 선택. 원격에 업로드되어 `image_node_id`에 바인딩됨 |
| `overrides` | STRING (입력) | "" | 선택. JSON `{node_id: <노드 전체 dict>}`. 각 항목이 **노드 전체를 교체**하며 마지막에 적용됨 |

| 출력 | 설명 |
|------|------|
| `output` | 디코드된 이미지/프레임 텐서 (애니메이션 출력은 프레임으로 펼쳐짐) |

#### Api Submit

입력은 **Api Generate**와 동일하며(`timeout_sec` 제외), 선택 `label`이 추가됩니다. OUTPUT_NODE라서 출력을 소비하는 노드가 없어도 실행됩니다.

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `label` | STRING | "" | 선택. 잡과 함께 기록되는 라벨 (이후 Api Collect가 반환) |

| 출력 | 설명 |
|------|------|
| `job_id` | 제출된 잡 id (이미 진행 중인 잡이 있으면 빈 문자열) |

#### Api Collect

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `wait_sec` | INT | 0 | `0`이면 준비됐을 때만 수집하고 아니면 즉시 건너뜀. 그 외엔 이 시간(초)까지 대기 (0–36000) |
| `poll_interval` | FLOAT | 2.0 | 대기 중 폴링 간격(초) (0.5–60.0) |

| 출력 | 설명 |
|------|------|
| `output` | 준비된 프레임. 없으면 다운스트림을 건너뛰는 `ExecutionBlocker` |
| `label` | 제출 시 기록된 라벨 |

> **종류별 단일 진행 잡:** API와 [Grok](#grok-노드-alcheminepackgrok) 노드는 패키지 디렉터리의 단일 `jobs.lock`을 공유하지만 종류별로 슬롯이 분리됩니다 — API 잡과 Grok 잡이 동시에 진행될 수 있고, 각 종류는 하나만 허용됩니다. 해당 종류의 잡이 이미 진행 중이면 Submit은 건너뛰고, Collect가 완료되면 슬롯을 비웁니다. `/loop` 등으로 Collect를 반복 실행하면 완료된 결과를 받아올 수 있습니다.

---

## 와일드카드 지원

`FilterTags`와 `ProcessTags` 노드는 `resources/wildcards.yaml`에 정의된 와일드카드를 지원합니다.

**예시:** 블랙리스트에 `__color__`를 사용하면 YAML 파일에 정의된 모든 색상(`red`, `blue`, `green` 등)에 매칭됩니다.

## 설정

> **`.env`는 선택 사항입니다** — 없어도 팩은 항상 정상 로드됩니다. [`.env.example`](.env.example)을 `.env`로 복사한 뒤 필요한 변수만 채우세요 (또는 같은 값을 노드 입력으로 전달). 자격증명이 필요한 노드가 값을 못 찾으면 **실행 시점에** 명확한 에러(ComfyUI 에러 창)를 띄웁니다 — 로딩 단계에선 절대 죽지 않습니다.

### `config.json` (OpenAI Inference 기본값)

이 패키지 루트에 `config.json`을 생성하여 **OpenAI Inference** 노드의 기본 `base_url`/`api_key`를 지정합니다 (노드 입력을 비워 두면 사용됨):

```json
{
  "inference": {
    "openai_base_url": "https://api.openai.com/v1",
    "openai_api_key": "your-api-key"
  }
}
```

### Grok 자격증명 (`.env` 또는 노드 입력)

**Grok Generate** 노드는 자격증명을 노드 입력에서 먼저 읽고, 입력이 비어 있으면 아래 `.env` 변수로 대체합니다:

```
GROK_ACCESS_TOKEN=...
GROK_REFRESH_TOKEN=...
GROK_CLIENT_ID=...
```

access token은 401/403에서 자동 갱신됩니다. `Grok token refresh failed (...)` 에러가 뜨면 refresh token이나 client_id가 만료/폐기된 것이니, x.ai에서 다시 인증해 값을 갱신하세요.

### Webshare 프록시 (선택, Danbooru)

`.env`에 `WEBSHARE_PROXY_USERNAME` / `WEBSHARE_PROXY_PASSWORD`를 넣으면 Danbooru 노드가 프록시를 경유합니다. 비워 두면 직결합니다.

## 예시

### ProcessTags 예시

```
입력: dog, cat, white dog, black cat
블랙리스트: cat
출력: white dog, black cat
필터됨: dog, cat
```

### FilterSubtags 예시

```
입력: dog, cat, white dog, black cat
출력: white dog, black cat
('dog'와 'cat'이 'white dog'와 'black cat'의 서브태그이므로 제거됨)
```

### SubstituteTags 예시

```
# "girl"이 없으면 "1boy"를 "1girl, 1boy"로 교체
pattern: 1boy
repl: 1girl, 1boy
skip_if: girl
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

## 라이선스

GPL-3.0 License
