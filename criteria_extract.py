"""
2단계: SBT 기여도 산정에 쓰이는 변수(카테고리·기간·지역·역할)를
원문 근거 인용과 함께 추출.

역할/카테고리의 "계수"는 LLM이 직접 정하지 않는다.
LLM은 텍스트에서 역할/카테고리 명칭만 근거와 함께 뽑고,
실제 계수는 로컬 매핑 테이블(ROLE_COEFFICIENT/CATEGORY_WEIGHT)에서
결정론적으로 조회한다 -> 같은 입력엔 항상 같은 계수, 감사 가능.
"""
import os
import json
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ---------- 로컬 계수 테이블 (실제 정책값으로 교체 필요) ----------
ROLE_COEFFICIENT = {
    "리더": 1.5, "팀장": 1.5, "주최자": 1.5, "기획자": 1.4,
    "참여자": 1.0, "봉사자": 1.0, "회원": 1.0,
    "보조": 0.8, "서포터": 0.8, "단순참가": 0.7,
}

CATEGORY_WEIGHT = {
    "봉사활동": 1.0,
    "지역행사참여": 0.8,
    "교육/멘토링": 1.2,
    "환경정화": 1.0,
    "지역경제활동": 1.1,
}

SYSTEM_PROMPT = """너는 지역 기여 활동 증빙 서류에서 SBT 점수 산정에 필요한 변수를 뽑는 어시스턴트다.

규칙:
1. 아래 스키마의 각 필드마다 "value"와 "evidence"를 함께 채워라.
2. "evidence"는 원문 텍스트에서 그 값을 판단한 근거가 되는 부분을 그대로 인용해야 한다 (지어내지 말 것).
3. 원문에서 확인할 수 없는 필드는 value를 null로, evidence도 null로 남겨라. 추측 금지.
4. role, category는 아래 후보 중 원문 내용과 가장 가까운 것 하나를 골라라. 후보에 없으면 원문 표현 그대로 적어라.
   - role 후보: 리더, 팀장, 주최자, 기획자, 참여자, 봉사자, 회원, 보조, 서포터, 단순참가
   - category 후보: 봉사활동, 지역행사참여, 교육/멘토링, 환경정화, 지역경제활동
5. 출력은 오직 JSON 객체 하나만. 설명, 코드블록 표시 금지.

스키마:
{
  "category": {"value": str|null, "evidence": str|null},
  "period_start": {"value": "YYYY-MM-DD"|null, "evidence": str|null},
  "period_end": {"value": "YYYY-MM-DD"|null, "evidence": str|null},
  "region": {"value": str|null, "evidence": str|null},
  "role": {"value": str|null, "evidence": str|null}
}
"""


def extract_criteria(ocr_result: dict, model: str = "claude-sonnet-4-6") -> dict:
    """OCR 결과(dict)를 받아 산정 변수 + 근거를 추출."""
    message = client.messages.create(
        model=model,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"OCR 원문:\n{ocr_result['full_text']}"}],
    )

    raw = "".join(b.text for b in message.content if b.type == "text").strip()
    cleaned = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM 응답 JSON 파싱 실패:\n{raw}") from e

    return _apply_local_mapping(parsed)


def _apply_local_mapping(parsed: dict) -> dict:
    """role/category 텍스트를 로컬 테이블로 계수 매핑. LLM이 숫자를 정하지 않게 함."""
    role_value = (parsed.get("role") or {}).get("value")
    category_value = (parsed.get("category") or {}).get("value")

    role_coefficient = ROLE_COEFFICIENT.get(role_value)
    category_weight = CATEGORY_WEIGHT.get(category_value)

    parsed["role"]["coefficient"] = role_coefficient
    parsed["role"]["mapped"] = role_coefficient is not None
    parsed["category"]["weight"] = category_weight
    parsed["category"]["mapped"] = category_weight is not None

    return parsed


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python criteria_extract.py <ocr_result.json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        ocr_result = json.load(f)

    result = extract_criteria(ocr_result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
