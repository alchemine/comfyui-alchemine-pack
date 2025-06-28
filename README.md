# ComfyUI-Alchemine-Pack

ComfyUI-Alchemine-Pack is a custom node pack for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
It provides nodes to assist with various workflow tasks, such as prompt processing.

## Provided Nodes

### ProcessTags
- **Description:**
  - Refines tags in prompts, removes tags based on a blacklist, removes subtags, replaces underscores, and offers various normalization options.
  - Multiple refinement options can be applied at once.
- **Category:** `AlcheminePack/Prompt`
- **Inputs:**
  - `text` (string)
  - `blacklist` (string, list of tags to remove)
  - `filter_subtags` (boolean)
  - `filter_tags` (boolean)
  - `replace_underscores` (boolean)
- **Outputs:**
  - `processed_text` (refined string)
  - `filtered_tags` (list of removed tags)
- **Example:**
  - Input: `dog, cat, white dog, black cat`, blacklist: `cat`, filter_tags: True, filter_subtags: True, replace_underscores: False
  - Output: `white dog, black cat`, removed tags: `cat`

### FilterTags
- **Description:**
  - Removes tags from prompts that match the blacklist. Recognizes tags with various bracket/weight notations.
- **Category:** `AlcheminePack/Prompt`
- **Inputs:**
  - `text` (string)
  - `blacklist` (string, list of tags to remove)
- **Outputs:**
  - `processed_text` (refined string)
  - `filtered_tags` (list of removed tags)
- **Example:**
  - Input: `dog, cat, white dog, black cat`, blacklist: `cat`
  - Output: `dog, white dog, black cat`, removed tags: `cat`

### FilterSubtags
- **Description:**
  - Removes duplicate or unnecessary subtags in prompts and normalizes various bracket/nesting/weight notations.
- **Category:** `AlcheminePack/Prompt`
- **Input:** `text` (string)
- **Outputs:**
  - `processed_text` (refined string)
  - `filtered_tags` (list of removed subtags)
- **Example:**
  - Input: `dog, cat, white dog, black cat`
  - Output: `white dog, black cat`
  - Input: `(cat:0.9), (cat:1.1), black cat, (black cat)`
  - Output: `(cat:0.9), (cat:1.1), black cat, (black cat)`

### FilterUnderscores
- **Description:**
  - Converts all underscores (_) in the prompt to spaces.
- **Category:** `AlcheminePack/Prompt`
- **Input:** `text` (string)
- **Output:**
  - `processed_text` (string with underscores removed)
- **Example:**
  - Input: `dog_cat_white_dog_black_cat`
  - Output: `dog cat white dog black cat`

## Installation & Usage

1. Copy or clone this repository into the `custom_nodes` directory of your ComfyUI installation.
2. Restart ComfyUI. You will then be able to use each node from the `AlcheminePack/Prompt` category in your workflow.

## Examples

Below are usage examples for each node.

```
[ProcessTags]
Input: dog, cat, white dog, black cat (blacklist: cat)
Output: white dog, black cat (removed tags: cat)

[filterTags]
Input: dog, cat, white dog, black cat (blacklist: cat)
Output: dog, white dog, black cat (removed tags: cat)

[filterSubtags]
Input: dog, cat, white dog, black cat
Output: white dog, black cat

Input: (cat:0.9), (cat:1.1), black cat, (black cat)
Output: (cat:0.9), (cat:1.1), black cat, (black cat)

[filterUnderscores]
Input: dog_cat_white_dog_black_cat
Output: dog cat white dog black cat 
```

---

# ComfyUI-Alchemine-Pack

ComfyUI-Alchemine-Pack은 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)를 위한 커스텀 노드 팩입니다.
프롬프트 처리 등 다양한 워크플로우를 보조하는 노드를 제공합니다.

## 제공 노드

### ProcessTags
- **설명:**
  - 프롬프트에서 태그를 정제하고, 블랙리스트 기반 태그 제거, 서브태그 제거, 언더스코어 제거 등 다양한 정규화 옵션을 제공합니다.
  - 여러 정제 옵션을 한 번에 적용할 수 있습니다.
- **카테고리:** `AlcheminePack/Prompt`
- **입력:**
  - `text` (문자열)
  - `blacklist` (문자열, 제거할 태그 목록)
  - `filter_subtags` (boolean)
  - `filter_tags` (boolean)
  - `replace_underscores` (boolean)
- **출력:**
  - `processed_text` (정제된 문자열)
  - `filtered_tags` (제거된 태그 목록)
- **예시:**
  - 입력: `dog, cat, white dog, black cat`, blacklist: `cat`, filter_tags: True, filter_subtags: True, replace_underscores: False
  - 출력: `white dog, black cat`, 제거된 태그: `cat`

### FilterTags
- **설명:**
  - 프롬프트에서 블랙리스트에 해당하는 태그를 제거합니다. 다양한 괄호/가중치 표기법을 정규화하여 태그를 인식합니다.
- **카테고리:** `AlcheminePack/Prompt`
- **입력:**
  - `text` (문자열)
  - `blacklist` (문자열, 제거할 태그 목록)
- **출력:**
  - `processed_text` (정제된 문자열)
  - `filtered_tags` (제거된 태그 목록)
- **예시:**
  - 입력: `dog, cat, white dog, black cat`, blacklist: `cat`
  - 출력: `dog, white dog, black cat`, 제거된 태그: `cat`

### FilterSubtags
- **설명:**
  - 프롬프트 내에서 중복되거나 불필요한 서브태그를 제거하고, 다양한 괄호/중첩/가중치 표기법을 정규화합니다.
- **카테고리:** `AlcheminePack/Prompt`
- **입력:** `text` (문자열)
- **출력:**
  - `processed_text` (정제된 문자열)
  - `filtered_tags` (제거된 서브태그 목록)
- **예시:**
  - 입력: `dog, cat, white dog, black cat`
  - 출력: `white dog, black cat`
  - 입력: `(cat:0.9), (cat:1.1), black cat, (black cat)`
  - 출력: `(cat:0.9), (cat:1.1), black cat, (black cat)`

### FilterUnderscores
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

[filterTags]
입력: dog, cat, white dog, black cat (blacklist: cat)
출력: dog, white dog, black cat (제거된 태그: cat)

[filterSubtags]
입력: dog, cat, white dog, black cat
출력: white dog, black cat

입력: (cat:0.9), (cat:1.1), black cat, (black cat)
출력: (cat:0.9), (cat:1.1), black cat, (black cat)

[filterUnderscores]
입력: dog_cat_white_dog_black_cat
출력: dog cat white dog black cat
```
