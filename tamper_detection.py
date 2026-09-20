"""
자격증 위변조 의심 여부를 점검하는 모듈.
- ELA(Error Level Analysis): 로컬, 토큰 0
- 메타데이터 검사: 로컬, 토큰 0
- 복사-붙여넣기(copy-move) 탐지: 로컬, 토큰 0
- 포맷/패턴 검증: 로컬, 토큰 0
- LLM 시각 이상 탐지: API 1회만 호출, 이미지 축소 + 짧은 응답으로 토큰 최소화
"""
import os
import io
import re
import base64
from datetime import datetime

import numpy as np
from PIL import Image
import cv2
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ---------- 1. ELA (Error Level Analysis) ----------
def run_ela(image_path: str, quality: int = 90) -> dict:
    """
    JPEG로 재압축했을 때 원본과의 차이를 분석.
    합성/편집된 영역은 압축 오차가 주변과 다르게 나타나는 경향이 있음.
    PNG 스캔본처럼 원래 무손실인 이미지는 이 검사의 신뢰도가 낮아짐 -> note로 표시.
    """
    img = Image.open(image_path).convert("RGB")
    is_lossy_original = image_path.lower().endswith((".jpg", ".jpeg"))

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)

    diff = np.array(img).astype(int) - np.array(resaved).astype(int)
    diff_abs = np.abs(diff)

    max_diff = int(diff_abs.max())
    mean_diff = float(diff_abs.mean())

    # 국소적으로 튀는 영역이 있는지 (표준편차 대비 최대값이 비정상적으로 큰 경우)
    std_diff = float(diff_abs.std()) or 1e-6
    spike_ratio = max_diff / std_diff

    suspicious = spike_ratio > 15 and is_lossy_original

    return {
        "applicable": is_lossy_original,
        "mean_diff": round(mean_diff, 3),
        "max_diff": max_diff,
        "spike_ratio": round(spike_ratio, 2),
        "suspicious": suspicious,
        "note": "PNG/무손실 원본이라 ELA 신뢰도 낮음" if not is_lossy_original else "",
    }


# ---------- 2. 메타데이터 검사 ----------
EDITING_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "paint.net", "illustrator", "snapseed",
    "picsart", "canva", "lightroom",
]


def check_metadata(image_path: str) -> dict:
    """EXIF에 편집 프로그램 흔적이 있는지, 생성/수정 시간이 부자연스러운지 확인."""
    img = Image.open(image_path)
    exif = img.getexif()

    software = None
    datetime_original = None
    datetime_modified = None

    if exif:
        software = exif.get(0x0131)  # Software tag
        datetime_original = exif.get(0x9003)  # DateTimeOriginal
        datetime_modified = exif.get(0x0132)  # DateTime (modify date)

    software_flag = False
    if software:
        software_lower = str(software).lower()
        software_flag = any(kw in software_lower for kw in EDITING_SOFTWARE_KEYWORDS)

    time_mismatch = False
    if datetime_original and datetime_modified and datetime_original != datetime_modified:
        time_mismatch = True

    return {
        "has_exif": bool(exif),
        "software": software,
        "software_flag": software_flag,
        "datetime_original": datetime_original,
        "datetime_modified": datetime_modified,
        "time_mismatch": time_mismatch,
        "note": "EXIF 없음: 스캔/캡처 과정에서 원래 제거됐을 수 있어 이 항목만으로 판단 불가" if not exif else "",
    }


# ---------- 3. 복사-붙여넣기(copy-move) 탐지 ----------
def detect_copy_move(image_path: str, block_size: int = 16, hash_size: int = 8) -> dict:
    """
    이미지를 블록으로 나눠 각 블록의 perceptual hash를 비교.
    서로 떨어진 위치에 거의 동일한 블록이 반복되면 복사-붙여넣기 의심.
    (도장, 워터마크처럼 원래 반복되는 패턴은 오탐 가능 -> 참고용 지표로만 사용)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    blocks = {}
    suspicious_pairs = 0

    step = block_size
    for y in range(0, h - block_size, step):
        for x in range(0, w - block_size, step):
            block = img[y:y + block_size, x:x + block_size]
            small = cv2.resize(block, (hash_size, hash_size))
            avg = small.mean()
            bits = (small > avg).flatten()
            phash = tuple(bits.astype(int))

            if phash in blocks:
                prev_positions = blocks[phash]
                for (py, px) in prev_positions:
                    dist = ((py - y) ** 2 + (px - x) ** 2) ** 0.5
                    if dist > block_size * 3:  # 인접 블록이 아닌 먼 곳에서 중복
                        suspicious_pairs += 1
                blocks[phash].append((y, x))
            else:
                blocks[phash] = [(y, x)]

    return {
        "suspicious_pairs": suspicious_pairs,
        "suspicious": suspicious_pairs > 5,
        "note": "도장/로고처럼 원래 반복되는 패턴에서는 오탐 가능",
    }


# ---------- 4. 포맷/패턴 검증 (규칙 기반, 토큰 0) ----------
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_format(structured: dict) -> dict:
    """LLM이 뽑은 JSON 필드가 상식적인 형식/범위인지 규칙으로 검증."""
    issues = []

    issue_date = structured.get("issue_date")
    if issue_date:
        if not DATE_PATTERN.match(str(issue_date)):
            issues.append(f"발급일 형식 비정상: {issue_date}")
        else:
            try:
                d = datetime.strptime(issue_date, "%Y-%m-%d")
                if d > datetime.now():
                    issues.append(f"발급일이 미래 날짜: {issue_date}")
                if d.year < 1980:
                    issues.append(f"발급일이 비정상적으로 오래됨: {issue_date}")
            except ValueError:
                issues.append(f"발급일 파싱 실패: {issue_date}")

    cert_number = structured.get("cert_number")
    if cert_number and len(str(cert_number)) < 4:
        issues.append(f"자격번호가 지나치게 짧음: {cert_number}")

    holder_name = structured.get("holder_name")
    if holder_name and not re.match(r"^[가-힣a-zA-Z\s]{2,20}$", str(holder_name)):
        issues.append(f"이름 형식 비정상: {holder_name}")

    return {"issues": issues, "suspicious": len(issues) > 0}


# ---------- 5. LLM 시각 이상 탐지 (API 1회, 이미지 축소) ----------
def _downscale_to_base64(image_path: str, max_side: int = 768) -> tuple[str, str]:
    """토큰 절약을 위해 이미지를 축소 후 base64 인코딩."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)))

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return b64, "image/jpeg"


VLM_SYSTEM_PROMPT = """자격증 이미지에서 위변조 의심 징후만 짧게 판단해.
확인할 것: 폰트 불일치, 글자 정렬 어긋남, 배경과 텍스트 경계 부자연스러움, 도장/직인 위치 이상.
반드시 아래 JSON 형식으로만, 다른 설명 없이 답해:
{"suspicious": true/false, "reason": "한 문장 이내"}"""


def vlm_visual_check(image_path: str, model: str = "claude-sonnet-4-6") -> dict:
    """이미지를 축소해서 1회만 호출. 응답도 짧게 강제해서 토큰 최소화."""
    b64, media_type = _downscale_to_base64(image_path)

    message = client.messages.create(
        model=model,
        max_tokens=100,  # 짧은 JSON만 필요하므로 낮게 제한
        system=VLM_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": "위 자격증을 확인해줘."},
            ],
        }],
    )

    raw = "".join(b.text for b in message.content if b.type == "text").strip()
    cleaned = raw.replace("```json", "").replace("```", "").strip()

    try:
        import json
        return json.loads(cleaned)
    except Exception:
        return {"suspicious": None, "reason": f"파싱 실패, 원본 응답: {raw}"}


# ---------- 통합 ----------
def run_tamper_check(image_path: str, structured: dict, skip_vlm_if_clean: bool = True) -> dict:
    """
    전체 위변조 검사 실행.
    skip_vlm_if_clean=True면 로컬 검사(1~4번)에서 아무 의심 신호도 없을 때
    LLM 호출(5번) 자체를 생략해서 토큰을 더 아낀다.
    """
    ela_result = run_ela(image_path)
    metadata_result = check_metadata(image_path)
    copy_move_result = detect_copy_move(image_path)
    format_result = validate_format(structured)

    local_suspicious = any([
        ela_result["suspicious"],
        metadata_result["software_flag"],
        metadata_result["time_mismatch"],
        copy_move_result["suspicious"],
        format_result["suspicious"],
    ])

    report = {
        "ela": ela_result,
        "metadata": metadata_result,
        "copy_move": copy_move_result,
        "format_validation": format_result,
        "vlm_check": None,
    }

    if skip_vlm_if_clean and not local_suspicious:
        report["vlm_check"] = {"skipped": True, "reason": "로컬 검사에서 의심 신호 없어 API 호출 생략"}
    else:
        report["vlm_check"] = vlm_visual_check(image_path)

    overall_suspicious = local_suspicious or bool(report["vlm_check"].get("suspicious"))
    report["overall_suspicious"] = overall_suspicious

    return report


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("사용법: python tamper_detection.py <이미지경로> [구조화된_json경로]")
        sys.exit(1)

    structured = {}
    if len(sys.argv) >= 3:
        with open(sys.argv[2], "r", encoding="utf-8") as f:
            structured = json.load(f)

    result = run_tamper_check(sys.argv[1], structured)
    print(json.dumps(result, ensure_ascii=False, indent=2))
