# 자격증 OCR -> 위변조 검사 -> 산정변수 추출 파이프라인 일괄 실행 스크립트
# 사용법: PowerShell에서 .\run.ps1 실행

$ErrorActionPreference = "Stop"

# 1. conda 환경 생성 (이미 있으면 이 줄에서 에러 나도 무시하고 계속 진행됨)
conda create -n cert-ocr python=3.10 -y

# 2. 환경 활성화
conda activate cert-ocr

# 3. PaddlePaddle 설치 (CPU 기준 - GPU면 아래 줄로 교체)
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
# GPU(CUDA 11.8)면 위 줄 대신:
# pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# 4. 나머지 패키지 설치
pip install -r requirements.txt

# 5. API 키 입력 (매번 직접 입력 - 코드/파일에 하드코딩 금지)
$apiKey = Read-Host "ANTHROPIC_API_KEY를 입력하세요" -AsSecureString
$env:ANTHROPIC_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($apiKey)
)

# 6. 파이프라인 실행
python pipeline.py --dir ./certs --out-dir ./results

Write-Host "`n완료. 결과는 .\results 폴더에 저장되었습니다."