"""
OCR로 추출된 텍스트(+신뢰도 정보)를 LLM API에 보내
구조화된 JSON(자격증 필드)으로 변환하는 모듈.
"""
import os
import json
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """너는 한국어 자격증/증명서 OCR 결과를 정리하는 어시스턴트다.
아래 규칙을 반드시 지켜라:

1. 입력은 OCR로 추출한 텍스트이며 오탈자나 줄바꿈 오류가 있을 수 있다.
   문맥상 명백한 오탈자(예: "발금일" -> "발급일" 같은 조사/받침 오류)는 보정해도 된다.
2. 그러나 이름, 자격증번호, 날짜, 숫자 등 사실 정보는 절대로 "그럴듯하게" 지어내지 마라.
   원문에서 읽을 수 없거나 애매하면 반드시 null로 남겨라.
3. OCR 신뢰도(confidence)가 낮게 표시된 라인의 내용은 특히 보수적으로 판단하고,
   확신이 없으면 해당 필드를 null 처리하라.
4. 출력은 오직 JSON 객체 하나만 반환한다. 설명, 코드블록 표시(```), 다른 텍스트를 절대 포함하지 마라.
5. 아래 스키마를 따르되, 문서에 해당 정보가 아예 없으면 필드 값은 null로 채운다.

스키마:
{
  "cert_name": "자격증/증명서 명칭",
  "holder_name": "취득자/대상자 이름",
  "issuer": "발급 기관명",
  "issue_date": "발급일 (YYYY-MM-DD 형식으로 정규화, 확실하지 않으면 원문 그대로)",
  "cert_number": "자격/등록 번호",
  "expiry_date": "유효기간 (있는 경우, 없으면 null)",
  "notes": "위 필드에 포함 안 된 특이사항이나 저신뢰도로 판단이 애매했던 부분 메모"
}
"""


def build_user_prompt(ocr_result: dict) -> str:
    lines_with_conf = "\n".join(
        f"[신뢰도 {l['confidence']:.2f}] {l['text']}" for l in ocr_result["lines"]
    )
    return f"""다음은 자격증 이미지를 OCR한 결과다. 각 줄 앞에 OCR 신뢰도를 표시했다.

{lines_with_conf}

위 규칙에 따라 JSON으로 정리해줘."""


def extract_json_from_text(ocr_result: dict, model: str = "claude-sonnet-4-6") -> dict:
    """
    OCR 결과(dict, ocr_extract.extract_text의 반환값)를 받아
    구조화된 자격증 정보 JSON(dict)을 반환.
    """
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(ocr_result)}],
    )

    raw_text = "".join(
        block.text for block in message.content if block.type == "text"
    ).strip()

    # 혹시 모델이 코드블록으로 감싸서 응답한 경우 방어적으로 제거
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM 응답을 JSON으로 파싱하지 못했습니다.\n원본 응답:\n{raw_text}"
        ) from e


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python llm_extract.py <ocr_result.json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        ocr_result = json.load(f)

    result = extract_json_from_text(ocr_result)
    print(json.dumps(result, ensure_ascii=False, indent=2))