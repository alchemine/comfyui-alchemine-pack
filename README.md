# ComfyUI-Alchemine-Pack

ComfyUI-Alchemine-Pack은 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)를 위한 커스텀 노드 팩입니다.
프롬프트 처리 및 태그 정규화 등 다양한 워크플로우를 보조하는 노드를 제공합니다.

## 제공 노드

### RemoveSubtags
- **설명:**
  - 프롬프트 내에서 중복되거나 불필요한 서브태그를 제거하고, 다양한 괄호/중첩/가중치 표기법을 정규화합니다.
  - 예시 입력/출력:
    - 입력: `dog, cat, white dog, black cat`
    - 출력: `white dog, black cat`
    - 입력: `(cat:0.9), (cat:1.1), black cat, (black cat)`
    - 출력: `(cat:0.9), (cat:1.1), black cat, (black cat)`
- **카테고리:** `AlcheminePack/Prompt`
- **입력:** `text` (문자열)
- **출력:** 정제된 문자열

## 설치 및 사용법

1. 이 저장소를 ComfyUI의 `custom_nodes` 디렉터리에 복사 또는 클론합니다.
2. ComfyUI를 재시작하면, 워크플로우 내에서 `AlcheminePack/Prompt` 카테고리에서 `Remove Subtags` 노드를 사용할 수 있습니다.

## 예시

아래는 RemoveSubtags 노드의 사용 예시입니다.

```
입력: dog, cat, white dog, black cat
출력: white dog, black cat

입력: (cat:0.9), (cat:1.1), black cat, (black cat)
출력: (cat:0.9), (cat:1.1), black cat, (black cat)
```

---

자세한 내용 및 업데이트는 [GitHub 저장소](https://github.com/alchemine/comfyui-alchemine-pack)를 참고하세요.
