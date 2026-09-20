"""
1단계 위변조 신호 + 2단계 산정변수 추출 결과를 점수화하고
최종 판정(자동통과/검토/반려)을 내리는 모듈.

핵심 원칙: AI는 '권고'만 한다. requires_human_review는 항상 True.
모든 점수 산출 근거(breakdown)를 함께 반환해서 사람이 왜 이 점수가
나왔는지 그대로 확인할 수 있게 한다 (설명 가능한 구조).
"""

AUTO_PASS_THRESHOLD = 0.7
REVIEW_THRESHOLD = 0.4


# ---------- 1단계: 위변조 점수화 ----------
def _score_ela(ela: dict) -> tuple[float, str]:
    if not ela["applicable"]:
        return 0.85, "PNG 등 무손실 원본이라 재압축 분석 불가 -> 중립값 부여, 사람 확인 권장"
    spike = ela["spike_ratio"]
    score = max(0.0, min(1.0, 1 - spike / 50))
    return round(score, 3), f"압축 오차 스파이크 비율 {spike} 기반 (높을수록 편집 의심)"


def _score_metadata(meta: dict) -> tuple[float, str]:
    score = 1.0
    reasons = []
    if meta["software_flag"]:
        score -= 0.4
        reasons.append(f"편집 프로그램 흔적 발견: {meta['software']}")
    if meta["time_mismatch"]:
        score -= 0.2
        reasons.append("촬영시각과 수정시각 불일치")
    if not meta["has_exif"]:
        reasons.append("EXIF 없음(스캔본일 수 있어 감점 없이 참고만)")
    return round(max(0.0, score), 3), ("; ".join(reasons) if reasons else "이상 신호 없음")


def _score_date_logic(format_result: dict) -> tuple[float, str]:
    date_issues = [i for i in format_result["issues"] if "발급일" in i]
    if not date_issues:
        return 1.0, "날짜 논리 이상 없음"
    score = max(0.0, 1 - 0.5 * len(date_issues))
    return round(score, 3), "; ".join(date_issues)


def compute_tamper_score(tamper_report: dict) -> dict:
    ela_score, ela_reason = _score_ela(tamper_report["ela"])
    meta_score, meta_reason = _score_metadata(tamper_report["metadata"])
    date_score, date_reason = _score_date_logic(tamper_report["format_validation"])

    cm = tamper_report["copy_move"]
    cm_score = 0.7 if cm["suspicious"] else 1.0
    cm_reason = f"의심 블록 쌍 {cm['suspicious_pairs']}개 발견" if cm["suspicious"] else "이상 없음"

    weights = {"ela": 0.35, "metadata": 0.25, "date_logic": 0.25, "copy_move": 0.15}
    total = (
        ela_score * weights["ela"]
        + meta_score * weights["metadata"]
        + date_score * weights["date_logic"]
        + cm_score * weights["copy_move"]
    )

    breakdown = {
        "ela": {"score": ela_score, "weight": weights["ela"], "reason": ela_reason},
        "metadata": {"score": meta_score, "weight": weights["metadata"], "reason": meta_reason},
        "date_logic": {"score": date_score, "weight": weights["date_logic"], "reason": date_reason},
        "copy_move": {"score": cm_score, "weight": weights["copy_move"], "reason": cm_reason},
    }

    vlm = tamper_report.get("vlm_check") or {}
    if vlm.get("suspicious") is True:
        capped = min(total, 0.5)
        breakdown["vlm_override"] = {
            "reason": vlm.get("reason"),
            "action": f"LLM 시각 검사에서 의심 신호 -> 점수 상한 0.5로 제한 (원래 {round(total,3)})",
        }
        total = capped
    elif vlm.get("skipped"):
        breakdown["vlm_override"] = {"reason": "로컬 검사 이상무로 생략됨", "action": "없음"}

    return {"score": round(total, 3), "breakdown": breakdown}


# ---------- 2단계: 추출 신뢰도 점수화 ----------
REQUIRED_FIELDS = ["category", "period_start", "region", "role"]


def compute_extraction_confidence(criteria: dict) -> dict:
    per_field = {}
    total = 0.0
    weight_each = 1.0 / len(REQUIRED_FIELDS)

    for field in REQUIRED_FIELDS:
        entry = criteria.get(field) or {}
        value = entry.get("value")
        evidence = entry.get("evidence")

        if value is None or evidence is None:
            per_field[field] = {"score": 0.0, "reason": "원문에서 근거를 찾지 못함"}
            continue

        # role/category는 로컬 테이블 매핑 성공 여부까지 반영
        if field == "role" and not entry.get("mapped"):
            per_field[field] = {"score": 0.5, "reason": f"역할 '{value}' 텍스트는 뽑았으나 계수 테이블에 없음 -> 사람이 계수 확인 필요"}
            total += weight_each * 0.5
            continue
        if field == "category" and not entry.get("mapped"):
            per_field[field] = {"score": 0.5, "reason": f"카테고리 '{value}' 텍스트는 뽑았으나 가중치 테이블에 없음 -> 사람이 확인 필요"}
            total += weight_each * 0.5
            continue

        per_field[field] = {"score": 1.0, "reason": f"근거 인용 확보: \"{evidence}\""}
        total += weight_each

    return {"score": round(total, 3), "breakdown": per_field}


# ---------- 최종 판정 ----------
def compute_final_decision(tamper_score_result: dict, extraction_confidence_result: dict) -> dict:
    tamper_score = tamper_score_result["score"]
    extraction_score = extraction_confidence_result["score"]

    overall = round(tamper_score * 0.6 + extraction_score * 0.4, 3)

    if overall >= AUTO_PASS_THRESHOLD:
        recommended = "자동통과 권고"
    elif overall >= REVIEW_THRESHOLD:
        recommended = "검토 필요"
    else:
        recommended = "반려 권고"

    return {
        "overall_score": overall,
        "tamper_score": tamper_score,
        "extraction_confidence_score": extraction_score,
        "recommended_decision": recommended,
        "requires_human_review": True,  # 항상 True — AI는 발급 여부를 단독 확정하지 않음
        "note": "이 판정은 참고용 권고이며, 최종 승인/반려는 반드시 사람이 확인 후 결정합니다.",
        "rationale": {
            "tamper": tamper_score_result["breakdown"],
            "extraction": extraction_confidence_result["breakdown"],
        },
    }
