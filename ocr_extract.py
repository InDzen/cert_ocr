"""
PaddleOCR(3.x) 이용한 텍스트 추출 모듈.
한글 자격증 이미지 -> (텍스트, 신뢰도) 리스트 반환.

주의: PaddleOCR 3.x부터 API가 크게 바뀜.
- use_angle_cls -> use_textline_orientation
- .ocr() -> .predict() 권장
- 결과가 OCRResult 객체(dict형)로 반환되며 'rec_texts', 'rec_scores' 키 사용
"""
from paddleocr import PaddleOCR
import cv2

_ocr_engine = None


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(
            lang="korean",
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    return _ocr_engine


def preprocess_image(image_path: str) -> str:
    """그레이스케일 + CLAHE 대비 향상. 필요할 때만 사용."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    tmp_path = image_path + ".preprocessed.png"
    cv2.imwrite(tmp_path, enhanced)
    return tmp_path


def extract_text(image_path: str, preprocess: bool = False) -> dict:
    """
    이미지에서 텍스트를 추출.
    preprocess 기본값 False -> 먼저 순정 상태로 zero-shot 테스트 권장.

    Returns:
        {"full_text": str, "lines": [{"text": str, "confidence": float}, ...]}
    """
    engine = get_ocr_engine()
    target_path = preprocess_image(image_path) if preprocess else image_path

    results = engine.predict(target_path)
    res = results[0]  # 이미지 1장 기준

    texts = res["rec_texts"]
    scores = res["rec_scores"]

    lines = [
        {"text": t, "confidence": round(float(s), 4)}
        for t, s in zip(texts, scores)
    ]
    full_text = "\n".join(l["text"] for l in lines)

    return {"full_text": full_text, "lines": lines}


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("사용법: python ocr_extract.py <이미지경로>")
        sys.exit(1)

    result = extract_text(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))