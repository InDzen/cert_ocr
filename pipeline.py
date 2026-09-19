"""
자격증 이미지 -> PaddleOCR -> LLM API -> JSON 저장까지의 전체 파이프라인.

사용법:
    python pipeline.py --image cert.jpg --out result.json
    python pipeline.py --dir ./certs --out-dir ./results   # 여러 장 일괄 처리
"""
import argparse
import json
import os
from pathlib import Path

from ocr_extract import extract_text
from llm_extract import extract_json_from_text


def process_single(image_path: str, save_ocr_debug: bool = True) -> dict:
    print(f"[OCR] {image_path} 처리 중...")
    ocr_result = extract_text(image_path)

    if save_ocr_debug:
        debug_path = str(Path(image_path).with_suffix(".ocr.json"))
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(ocr_result, f, ensure_ascii=False, indent=2)
        print(f"  -> OCR 원본 결과 저장: {debug_path}")

    print(f"[LLM] 구조화 추출 중...")
    structured = extract_json_from_text(ocr_result)
    structured["_source_image"] = os.path.basename(image_path)

    return structured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="단일 이미지 경로")
    parser.add_argument("--dir", help="여러 이미지가 있는 디렉토리 (일괄 처리)")
    parser.add_argument("--out", default="result.json", help="단일 처리 시 출력 파일")
    parser.add_argument("--out-dir", default="results", help="일괄 처리 시 출력 디렉토리")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("환경변수 ANTHROPIC_API_KEY가 설정되어 있지 않습니다.")

    if args.image:
        result = process_single(args.image)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n완료: {args.out}")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.dir:
        os.makedirs(args.out_dir, exist_ok=True)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = [p for p in Path(args.dir).iterdir() if p.suffix.lower() in exts]

        all_results = []
        for img_path in image_files:
            try:
                result = process_single(str(img_path), save_ocr_debug=False)
                out_path = Path(args.out_dir) / f"{img_path.stem}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                all_results.append(result)
                print(f"  -> 저장: {out_path}\n")
            except Exception as e:
                print(f"  !! 실패 ({img_path.name}): {e}\n")

        merged_path = Path(args.out_dir) / "_merged.json"
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"전체 {len(all_results)}건 처리 완료. 통합 결과: {merged_path}")

    else:
        parser.error("--image 또는 --dir 중 하나는 반드시 지정해야 합니다.")


if __name__ == "__main__":
    main()