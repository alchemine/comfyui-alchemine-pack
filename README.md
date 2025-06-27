# ComfyUI-Alchemine-Pack

ComfyUI-Alchemine-Pack은 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)를 위한 커스텀 노드 팩입니다.
프롬프트 처리 및 태그 정규화 등 다양한 워크플로우를 보조하는 노드를 제공합니다.

## 제공 노드

### ProcessTags
- **설명:**
  - 프롬프트에서 태그를 정제하고, 블랙리스트 기반 태그 제거, 서브태그 제거, 언더스코어 제거 등 다양한 정규화 옵션을 제공합니다.
  - 여러 정제 옵션을 한 번에 적용할 수 있습니다.
- **카테고리:** `AlcheminePack/Prompt`
- **입력:**
  - `text` (문자열)
  - `blacklist` (문자열, 제거할 태그 목록)
  - `remove_subtags` (boolean)
  - `remove_tags` (boolean)
  - `remove_underscores` (boolean)
- **출력:**
  - `processed_text` (정제된 문자열)
  - `removed_tags` (제거된 태그 목록)
- **예시:**
  - 입력: `dog, cat, white dog, black cat`, blacklist: `cat`, remove_tags: True, remove_subtags: True, remove_underscores: False
  - 출력: `white dog, black cat`, 제거된 태그: `cat`

### RemoveTags
- **설명:**
  - 프롬프트에서 블랙리스트에 해당하는 태그를 제거합니다. 다양한 괄호/가중치 표기법을 정규화하여 태그를 인식합니다.
- **카테고리:** `AlcheminePack/Prompt`
- **입력:**
  - `text` (문자열)
  - `blacklist` (문자열, 제거할 태그 목록)
- **출력:**
  - `processed_text` (정제된 문자열)
  - `removed_tags` (제거된 태그 목록)
- **예시:**
  - 입력: `dog, cat, white dog, black cat`, blacklist: `cat`
  - 출력: `dog, white dog, black cat`, 제거된 태그: `cat`

### RemoveSubtags
- **설명:**
  - 프롬프트 내에서 중복되거나 불필요한 서브태그를 제거하고, 다양한 괄호/중첩/가중치 표기법을 정규화합니다.
- **카테고리:** `AlcheminePack/Prompt`
- **입력:** `text` (문자열)
- **출력:**
  - `processed_text` (정제된 문자열)
  - `removed_tags` (제거된 서브태그 목록)
- **예시:**
  - 입력: `dog, cat, white dog, black cat`
  - 출력: `white dog, black cat`
  - 입력: `(cat:0.9), (cat:1.1), black cat, (black cat)`
  - 출력: `(cat:0.9), (cat:1.1), black cat, (black cat)`

### RemoveUnderscores
- **설명:**
  - 프롬프트 내 모든 언더스코어(_)를 공백으로 변환합니다.
- **카테고리:** `AlcheminePack/Prompt`
- **입력:** `text` (문자열)
- **출력:**
  - `processed_text` (언더스코어가 제거된 문자열)
- **예시:**
  - 입력: `dog_cat_white_dog_black_cat`
  - 출력: `dog cat white dog black cat`

## 설치 및 사용법

1. 이 저장소를 ComfyUI의 `custom_nodes` 디렉터리에 복사 또는 클론합니다.
2. ComfyUI를 재시작하면, 워크플로우 내에서 `AlcheminePack/Prompt` 카테고리에서 각 노드를 사용할 수 있습니다.

## 예시

아래는 각 노드의 사용 예시입니다.

```
[ProcessTags]
입력: dog, cat, white dog, black cat (blacklist: cat)
출력: white dog, black cat (제거된 태그: cat)

[RemoveTags]
입력: dog, cat, white dog, black cat (blacklist: cat)
출력: dog, white dog, black cat (제거된 태그: cat)

[RemoveSubtags]
입력: dog, cat, white dog, black cat
출력: white dog, black cat

입력: (cat:0.9), (cat:1.1), black cat, (black cat)
출력: (cat:0.9), (cat:1.1), black cat, (black cat)

[RemoveUnderscores]
입력: dog_cat_white_dog_black_cat
출력: dog cat white dog black cat
```

---

자세한 내용 및 업데이트는 [GitHub 저장소](https://github.com/alchemine/comfyui-alchemine-pack)를 참고하세요.
