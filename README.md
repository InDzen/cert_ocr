# 자격증 OCR → JSON 추출 파이프라인

PaddleOCR로 자격증 이미지에서 텍스트를 추출하고, Claude API(LLM)로 이름/발급기관/발급일 등 핵심 정보를 구조화된 JSON으로 정리하는 파이프라인입니다.

## 폴더 구성

```
cert-ocr-pipeline/
├── requirements.txt
├── ocr_extract.py     # PaddleOCR 텍스트 추출
├── llm_extract.py     # Claude API로 JSON 구조화
├── pipeline.py         # 전체 실행 (OCR -> LLM -> JSON 저장)
└── README.md
```

## ⚠️ API 키 관련 필독 사항

이 프로젝트는 **Anthropic Claude API 키**를 사용합니다. 코드 어디에도 API 키가 하드코딩되어 있지 않으며, 실행 전 **본인의 API 키를 직접 발급받아 터미널에 환경변수로 등록**해야 정상 동작합니다.

- API 키는 [console.anthropic.com](https://console.anthropic.com) 에서 개인적으로 발급받아야 합니다.
- 발급받은 키는 절대 코드나 README, 커밋 이력 등에 포함하지 마세요.
- 아래 "실행 방법"의 키 등록 명령어를 **매 터미널 세션마다 직접 입력**해야 합니다 (영구 저장하지 않는 이상 터미널을 새로 열 때마다 다시 설정 필요).

## 설치

```bash
conda create -n cert-ocr python=3.10 -y
conda activate cert-ocr

# CPU 환경
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
# GPU 환경(CUDA 11.8)이면 위 줄 대신:
# pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

pip install -r requirements.txt
```

## 실행 방법

### 1. API 키 발급
[console.anthropic.com](https://console.anthropic.com) 에서 회원가입 후 API 키(`sk-ant-...`)를 발급받습니다.

### 2. API 키를 터미널에 직접 입력

**Windows (PowerShell)**
```powershell
$env:ANTHROPIC_API_KEY="본인의_API_키를_여기에_입력"
```

**macOS / Linux**
```bash
export ANTHROPIC_API_KEY="본인의_API_키를_여기에_입력"
```

> 이 명령은 터미널을 새로 열 때마다 다시 입력해야 합니다. 키를 파일에 저장해두고 관리하고 싶다면 `.env` 파일을 만들어 사용할 수 있으나, 이 경우에도 `.env` 파일을 절대 외부에 공유하거나 커밋하지 마세요.

### 3. 파이프라인 실행

```bash
python pipeline.py --image cert.jpg --out result.json
```

여러 장 일괄 처리:
```bash
python pipeline.py --dir ./certs --out-dir ./results
```

## 결과 예시

```json
{
  "cert_name": "정보처리기사",
  "holder_name": "홍길동",
  "issuer": "한국산업인력공단",
  "issue_date": "2024-03-15",
  "cert_number": "24202012345A",
  "expiry_date": null,
  "notes": "발급일 옆 도장으로 일부 글자 가려짐",
  "_source_image": "cert.jpg"
}
```

## 참고

- OCR 원본 결과(`*.ocr.json`)도 함께 저장되어, 인식 신뢰도(confidence)를 확인해 정확도 문제를 디버깅할 수 있습니다.
- 자격증 양식이 다양하거나 OCR 정확도가 낮게 나오는 경우, `ocr_extract.py`의 `preprocess=True`로 전처리(대비 보정)를 켜서 재시도할 수 있습니다.
