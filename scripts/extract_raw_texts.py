from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

from pypdf import PdfReader


RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed/raw_texts")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_name(path: Path) -> str:
    name = path.name
    name = re.sub(r"[^\w가-힣._-]+", "_", name)
    return name + ".txt"


def extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            text = f"[PAGE {i} EXTRACT ERROR: {e}]"
        pages.append(f"\n\n--- PAGE {i} ---\n{text}")
    return "\n".join(pages)


def extract_hwpml_text(xml_text: str) -> str:
    # 국가법령정보센터 HWP가 HWPML/XML 형태일 때 대응
    xml_text = re.sub(r"<!DOCTYPE.*?\]>", "", xml_text, flags=re.DOTALL)
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        # XML 파싱 실패 시 태그 제거 fallback
        cleaned = re.sub(r"<[^>]+>", " ", xml_text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    texts = []
    for elem in root.iter():
        if elem.text and elem.text.strip():
            texts.append(elem.text.strip())
    return "\n".join(texts)


def extract_hwp(path: Path) -> str:
    # 1) 법제처 HWPML처럼 실제로 XML 텍스트인 경우
    raw = path.read_bytes()
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            text = raw.decode(enc)
            if "<HWPML" in text or "<?xml" in text:
                return extract_hwpml_text(text)
        except UnicodeDecodeError:
            pass

    # 2) 혹시 압축형이면 내부 텍스트 시도
    if zipfile.is_zipfile(path):
        out = []
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.lower().endswith((".xml", ".txt")):
                    try:
                        data = z.read(name).decode("utf-8", errors="ignore")
                        out.append(f"\n--- {name} ---\n{extract_hwpml_text(data)}")
                    except Exception as e:
                        out.append(f"\n--- {name} ERROR: {e} ---")
        return "\n".join(out)

    return "[HWP extraction failed: unsupported binary HWP format]"


def main():
    files = list(RAW_DIR.rglob("*"))
    targets = [p for p in files if p.is_file() and p.suffix.lower() in [".pdf", ".hwp"]]

    print(f"[INFO] found {len(targets)} files")

    for path in targets:
        print(f"[EXTRACT] {path}")
        if path.suffix.lower() == ".pdf":
            text = extract_pdf(path)
        elif path.suffix.lower() == ".hwp":
            text = extract_hwp(path)
        else:
            continue

        out = OUT_DIR / safe_name(path)
        out.write_text(text, encoding="utf-8")
        print(f"  -> {out} ({len(text):,} chars)")


if __name__ == "__main__":
    main()
