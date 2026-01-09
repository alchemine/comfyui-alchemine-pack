# ComfyUI-Alchemine-Pack

[ComfyUI](https://github.com/comfyanonymous/ComfyUI)를 위한 커스텀 노드 팩입니다. 프롬프트 처리, Danbooru 연동, LLM 추론, 워크플로우 제어 등 다양한 유틸리티 노드를 제공합니다.

## 설치 방법

1. 이 저장소를 ComfyUI의 `custom_nodes` 디렉터리에 클론하거나 복사합니다.
2. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```
3. (선택) Danbooru 노드 사용 시 Playwright 브라우저 설치:
   ```bash
   playwright install
   ```
4. ComfyUI를 재시작합니다.

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

#### FilterTags

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `text` | STRING | (필수) | 입력 프롬프트 텍스트 |
| `blacklist_tags` | STRING | "" | 쉼표로 구분된 블랙리스트 (와일드카드 지원) |
| `fixed_tags` | STRING | "" | 필터링에 관계없이 보존할 태그 |

#### SubstituteTags

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `text` | STRING | (필수) | 입력 프롬프트 텍스트 |
| `pattern` | STRING | "" | 매칭할 정규식 패턴 |
| `repl` | STRING | "" | 대체 문자열 |
| `run_if` | STRING | "" | 이 패턴이 있을 때만 실행 |
| `skip_if` | STRING | "" | 이 패턴이 있으면 건너뜀 |

---

### Danbooru 노드 (`AlcheminePack/Danbooru`)

![Danbooru Workflow](workflows/comfyui-alchemine-pack-workflow-Danbooru.png)

| 노드 | 설명 |
|------|------|
| **Danbooru Post Tags Retriever** | 포스트 ID로 특정 Danbooru 포스트의 태그를 가져옵니다. |
| **Danbooru Related Tags Retriever** | Danbooru에서 빈도/유사도 기반으로 관련 태그를 검색합니다. |
| **Danbooru Popular Posts Tags Retriever** | 인기 포스트(일간/주간/월간)에서 태그를 가져옵니다. |
| **Danbooru Posts Downloader** | 검색 태그 기반으로 Danbooru 이미지를 다운로드합니다. |

> ⚠️ **주의:** 과도한 요청은 차단될 수 있습니다. 캐싱과 요청 제한을 활용하세요.

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
| `random` | BOOLEAN | True | 랜덤 선택 |
| `seed` | INT | 0 | 랜덤 시드 |

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
| **Gemini Inference** | Google Gemini API로 텍스트 생성. 비전 및 씽킹 모드 지원. |
| **Ollama Inference** | 로컬 Ollama API로 텍스트 생성. 비전 모델 지원. |
| **Text Editing Inference** | CoEdit 모델을 사용한 문법 교정 및 텍스트 편집. |

#### Gemini Inference

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `system_instruction` | STRING | "You are a helpful assistant." | 시스템 프롬프트 |
| `prompt` | STRING | "Hello, world!" | 사용자 프롬프트 |
| `gemini_api_key` | STRING | "" | API 키 (`config.json`에 설정 가능) |
| `model` | STRING | "latest" | 모델명 (`latest`, `latest-flash-lite`, `latest-pro-preview` 등) |
| `max_output_tokens` | INT | 100 | 최대 출력 토큰 |
| `seed` | INT | 0 | 랜덤 시드 |
| `think` | BOOLEAN | False | 씽킹 모드 활성화 |
| `candidate_count` | INT | 1 | 후보 수 (1-8) |
| `image` | IMAGE | (선택) | 비전 작업용 입력 이미지 |

#### Ollama Inference

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `system_instruction` | STRING | "You are a helpful assistant." | 시스템 프롬프트 |
| `prompt` | STRING | "Hello, world!" | 사용자 프롬프트 |
| `ollama_url` | STRING | "" | Ollama API URL (`config.json`에 설정 가능) |
| `model` | STRING | "" | 모델명 (Ollama에서 사용 가능해야 함) |
| `max_output_tokens` | INT | 100 | 최대 출력 토큰 |
| `seed` | INT | 0 | 랜덤 시드 |
| `think` | BOOLEAN | False | 씽킹 모드 활성화 |
| `image` | IMAGE | (선택) | 비전 작업용 입력 이미지 |

#### Text Editing Inference

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `predefined_system_instruction` | ENUM | "Fix the grammar" | 사전 정의된 지시 |
| `system_instruction` | STRING | "" | 커스텀 지시 (설정 시 사전 정의 지시 대체) |
| `prompt` | STRING | (예시 텍스트) | 편집할 입력 텍스트 |
| `seed` | INT | 0 | 랜덤 시드 |

사용 가능한 사전 정의 지시:
- Fix the grammar (문법 수정)
- Make this text coherent (텍스트 일관성 있게)
- Rewrite to make this easier to understand (이해하기 쉽게 재작성)
- Paraphrase this (바꿔 말하기)
- Write this more formally (더 격식 있게)
- Write in a more neutral way (더 중립적으로)

---

### 입력 노드 (`AlcheminePack/Input`)

![Input Workflow](workflows/comfyui-alchemine-pack-workflow-Input.png)

| 노드 | 설명 |
|------|------|
| **Width Height** | 스왑과 스케일 옵션이 있는 너비/높이 설정 노드. |

#### Width Height

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `width` | INT | 512 | 너비 값 |
| `height` | INT | 512 | 높이 값 |
| `swap` | BOOLEAN | False | 너비와 높이 교환 |
| `scale` | FLOAT | 1.0 | 스케일 배율 |

---

### 플로우 컨트롤 노드 (`AlcheminePack/FlowControl`)

![Flow Control Workflow](workflows/comfyui-alchemine-pack-workflow-FlowControl.png)

| 노드 | 설명 |
|------|------|
| **Signal Switch** | `signal`이 수신된 후 `value`를 전달합니다. 실행 순서를 제어합니다. |

#### Signal Switch

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `signal` | ANY | 시그널 입력 (이 입력이 완료될 때까지 대기) |
| `value` | ANY | 전달할 값 |

**사용 사례:** 순차 실행이 필요할 때 (예: 생성 A가 완료된 후에만 생성 B 실행).

---

### IO 노드 (`AlcheminePack/IO`) *(실험적)*

| 노드 | 설명 |
|------|------|
| **AsyncSaveImage** | 스레딩을 사용하여 비동기적으로 이미지를 저장합니다. |
| **PreviewLatestImage** | 출력 디렉터리에서 최신 이미지를 로드합니다. |

---

### Lora 노드 (`AlcheminePack/Lora`) *(실험적)*

| 노드 | 설명 |
|------|------|
| **DownloadImage** | URL에서 이미지를 다운로드합니다. |
| **SaveImageWithText** | 이미지와 텍스트 파일을 함께 저장합니다 (학습 데이터셋용). |

---

## 와일드카드 지원

`FilterTags`와 `ProcessTags` 노드는 `resources/wildcards.yaml`에 정의된 와일드카드를 지원합니다.

**예시:** 블랙리스트에 `__color__`를 사용하면 YAML 파일에 정의된 모든 색상(`red`, `blue`, `green` 등)에 매칭됩니다.

## 설정

이 패키지 루트에 `config.json` 파일을 생성하여 API 키와 설정을 관리합니다:

```json
{
  "inference": {
    "gemini_api_key": "your-gemini-api-key",
    "ollama_url": "http://localhost:11434"
  }
}
```

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

## 라이선스

GPL-3.0 License
