from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


TARGET_NOTEBOOKS: list[Path] = [
    Path("/home/jik/projects/study/pytorch/docs/topics/00a_Math_00_Roadmap.ipynb"),
    Path("/home/jik/projects/study/pytorch/docs/topics/00a_Math_01_Quick_Start.ipynb"),
    Path("/home/jik/projects/study/pytorch/docs/topics/00a_Math_02_Linear_Algebra_Foundations.ipynb"),
    Path("/home/jik/projects/study/pytorch/docs/topics/00a_Math_03_Calculus_Foundations.ipynb"),
    Path("/home/jik/projects/study/pytorch/docs/topics/00a_Math_04_Probability_Statistics_01_Introduction.ipynb"),
    Path("/home/jik/projects/study/pytorch/docs/topics/00a_Math_04_Probability_Statistics_02a_Probability_Basics.ipynb"),
    Path("/home/jik/projects/study/pytorch/docs/topics/00a_Math_04_Probability_Statistics_02b_Probability_Distributions.ipynb"),
    Path("/home/jik/projects/study/pytorch/docs/topics/00a_Math_04_Probability_Statistics_02c_Information_Theory_MLE.ipynb"),
    Path("/home/jik/projects/study/pytorch/docs/topics/00a_Math_04_Probability_Statistics_02d_Loss_Functions.ipynb"),
    Path("/home/jik/projects/study/pytorch/docs/topics/00a_Math_04_Probability_Statistics_03_PyTorch_Implementation.ipynb"),
    Path("/home/jik/projects/study/pytorch/docs/topics/00a_Math_04_Probability_Statistics_04_Summary.ipynb"),
    Path("/home/jik/projects/study/pytorch/docs/topics/00b_Probability_Statistics_Foundations.ipynb"),
]

OUTPUT_DIR = Path("/home/jik/projects/study/pytorch/docs/math")


def load_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_markdown_cell(source: Iterable[str]) -> list[str]:
    text = "".join(source).rstrip("\n")
    return [text] if text else []


def format_code_cell(cell: dict) -> list[str]:
    source = "".join(cell.get("source", []))
    lines: list[str] = ["```python", source.rstrip("\n"), "```"]

    output_lines = extract_outputs(cell.get("outputs", []))
    if output_lines:
        lines.append("```text")
        lines.extend(output_lines)
        lines.append("```")

    return lines


def extract_outputs(outputs: Iterable[dict]) -> list[str]:
    captured: list[str] = []

    for output in outputs or []:
        otype = output.get("output_type")

        if otype == "stream":
            text = output.get("text", "")
            if isinstance(text, list):
                for item in text:
                    captured.extend(item.rstrip("\n").splitlines())
            elif isinstance(text, str) and text:
                captured.extend(text.rstrip("\n").splitlines())
        elif otype in {"execute_result", "display_data"}:
            data = output.get("data", {})
            text_data = data.get("text/plain")
            if isinstance(text_data, list):
                captured.extend([line.rstrip("\n") for line in text_data])
            elif isinstance(text_data, str):
                captured.extend(text_data.rstrip("\n").splitlines())
        elif otype == "error":
            traceback = output.get("traceback", [])
            captured.extend([line.rstrip("\n") for line in traceback])

    return captured


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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for notebook in TARGET_NOTEBOOKS:
        if not notebook.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook}")
        convert_notebook(notebook, OUTPUT_DIR)


if __name__ == "__main__":
    main()

