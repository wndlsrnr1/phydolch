from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence


SOURCE_ROOT = Path("/home/jik/study/pytorch/jupyter_notebook_ver")
SOURCE_DIR = SOURCE_ROOT / "pytorch"
EXTRA_NOTEBOOKS: Sequence[Path] = [SOURCE_ROOT / "파이토치.ipynb"]
OUTPUT_DIR = Path("/home/jik/study/pytorch/pytorch")


def load_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_markdown_cell(source: Iterable[str]) -> list[str]:
    text = "".join(source).rstrip("\n")
    return [text] if text else []


def format_code_cell(cell: dict) -> list[str]:
    source = "".join(cell.get("source", []))
    lines: list[str] = ["```python", source.rstrip("\n"), "```"]

    text_outputs, media_outputs = extract_outputs(cell.get("outputs", []))
    if text_outputs:
        lines.append("```text")
        lines.extend(text_outputs)
        lines.append("```")
    if media_outputs:
        lines.extend(media_outputs)

    return lines


def extract_outputs(outputs: Iterable[dict]) -> tuple[list[str], list[str]]:
    text_captured: list[str] = []
    media_captured: list[str] = []
    media_index = 1

    for output in outputs or []:
        otype = output.get("output_type")

        if otype == "stream":
            text = output.get("text", "")
            if isinstance(text, list):
                for item in text:
                    text_captured.extend(item.rstrip("\n").splitlines())
            elif isinstance(text, str) and text:
                text_captured.extend(text.rstrip("\n").splitlines())
        elif otype in {"execute_result", "display_data"}:
            data = output.get("data", {})
            text_data = data.get("text/plain")
            if isinstance(text_data, list):
                text_captured.extend([line.rstrip("\n") for line in text_data])
            elif isinstance(text_data, str):
                text_captured.extend(text_data.rstrip("\n").splitlines())

            image_data = None
            mime_type = None
            if "image/png" in data:
                image_data = data.get("image/png")
                mime_type = "image/png"
            elif "image/jpeg" in data:
                image_data = data.get("image/jpeg")
                mime_type = "image/jpeg"

            if image_data and mime_type:
                if isinstance(image_data, list):
                    payload = "".join(image_data)
                else:
                    payload = image_data
                payload = payload.strip()
                if payload:
                    media_captured.append(
                        f"![output_{media_index}]"
                        f"(data:{mime_type};base64,{payload})"
                    )
                    media_index += 1
        elif otype == "error":
            traceback = output.get("traceback", [])
            text_captured.extend([line.rstrip("\n") for line in traceback])

    return text_captured, media_captured


def notebook_to_markdown(nb_data: dict) -> str:
    lines: list[str] = []

    for cell in nb_data.get("cells", []):
        ctype = cell.get("cell_type")

        if ctype == "markdown":
            lines.extend(format_markdown_cell(cell.get("source", [])))
        elif ctype == "code":
            lines.extend(format_code_cell(cell))
        else:
            continue

        lines.append("")  # 셀 간 구분을 위한 빈 줄

    # 마지막에 추가된 공백 줄 제거 후 끝에 개행 추가
    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines) + "\n"


def convert_notebook(path: Path, output_dir: Path) -> Path:
    nb_data = load_notebook(path)
    markdown = notebook_to_markdown(nb_data)

    output_path = output_dir / (path.stem + ".md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def discover_notebooks() -> list[Path]:
    candidates: list[Path] = list(SOURCE_DIR.glob("*.ipynb"))
    for extra in EXTRA_NOTEBOOKS:
        if extra.exists() and extra.suffix == ".ipynb":
            candidates.append(extra)

    seen: set[str] = set()
    notebooks: list[Path] = []
    for notebook in sorted(candidates):
        identifier = notebook.resolve().as_posix()
        if identifier in seen:
            continue
        seen.add(identifier)
        notebooks.append(notebook)

    return notebooks


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    notebooks = discover_notebooks()
    if not notebooks:
        raise FileNotFoundError(
            f"No notebooks found in {SOURCE_DIR} or extras: {EXTRA_NOTEBOOKS}"
        )

    for notebook in notebooks:
        convert_notebook(notebook, OUTPUT_DIR)


if __name__ == "__main__":
    main()

