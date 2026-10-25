#!/usr/bin/env python
"""docs/REPORT.md -> native LaTeX source and report/report.pdf.

    python scripts/build_report.py [--max-pages 4]
The source conversion uses pypandoc when available and the PDF is compiled with XeLaTeX or LuaLaTeX.
Exit 1 if the PDF is longer than --max-pages.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = r"""\documentclass[9pt,a4paper]{extarticle}
\usepackage[margin=1.55cm]{geometry}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\setmonofont{Consolas}[Scale=0.86]
\usepackage{amsmath,amssymb}
\usepackage{booktabs,longtable,array,calc}
\usepackage{graphicx}
\setkeys{Gin}{keepaspectratio}
\usepackage[hidelinks]{hyperref}
\usepackage{float}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\setlength{\parskip}{3.2pt}
\setlength{\parindent}{0pt}
\setlength{\tabcolsep}{3.2pt}
\renewcommand{\arraystretch}{1.05}
\linespread{1.015}
\emergencystretch=1em
\setcounter{secnumdepth}{0}
\setcounter{tocdepth}{0}
\title{%(title)s}
\author{%(subtitle)s}
\date{%(date)s}
\begin{document}
\maketitle
\vspace{-1.8em}
%(body)s
\end{document}
"""


def find_engine() -> str | None:
    for name in ("xelatex", "lualatex"):
        found = shutil.which(name)
        if found:
            return found
    local = Path.home() / "AppData/Local/Programs/MiKTeX/miktex/bin/x64/xelatex.exe"
    return str(local) if local.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=4)
    a = ap.parse_args(argv)
    src = ROOT / "docs/REPORT.md"
    out = ROOT / "report"
    out.mkdir(exist_ok=True)
    text = src.read_text(encoding="utf-8")
    meta = dict(re.findall(r'^(title|subtitle|date):\s*"?(.*?)"?\s*$', text, flags=re.M))
    try:
        import pypandoc
        body = pypandoc.convert_file(
            str(src), "latex", format="markdown+tex_math_dollars",
            extra_args=[f"--resource-path={ROOT}"])
    except ImportError:
        body = subprocess.run(["pandoc", str(src), "-t", "latex", "--from", "markdown+tex_math_dollars"],
                              capture_output=True, text=True, check=True).stdout
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    body = body.replace("\\begin{figure}\n", "\\begin{figure}[H]\n")
    tex = WRAPPER % {"title": meta.get("title", "Report"), "subtitle": meta.get("subtitle", ""), "date": meta.get("date", ""), "body": body}
    with (out / "report.tex").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(tex)
    engine = find_engine()
    if not engine:
        sys.exit("No XeLaTeX or LuaLaTeX engine found; report/report.tex was generated but not compiled")
    (out / "report.pdf").unlink(missing_ok=True)      # a failed build must never leave a stale PDF behind
    for _ in range(2):
        r = subprocess.run([engine, "--enable-installer", "-interaction=nonstopmode", "-halt-on-error",
                            "-output-directory=report", "report/report.tex"], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not (out / "report.pdf").exists():
        print(r.stdout[-3000:])
        sys.exit(f"LaTeX failed (exit {r.returncode})")
    if "\\usepackage" in body:
        sys.exit("pandoc emitted \\usepackage into the body: the YAML header was not recognised (blank line before it?)")
    for ext in ("aux", "log", "out"):
        (out / f"report.{ext}").unlink(missing_ok=True)
    pages = None
    if shutil.which("pdfinfo"):
        info = subprocess.run(["pdfinfo", str(out / "report.pdf")], capture_output=True, text=True).stdout
        m = re.search(r"Pages:\s+(\d+)", info)
        pages = int(m.group(1)) if m else None
    if pages is None:
        try:
            from pypdf import PdfReader  # type: ignore
            pages = len(PdfReader(str(out / "report.pdf")).pages)
        except Exception:  # noqa: BLE001
            data = (out / "report.pdf").read_bytes()
            pages = len(re.findall(rb"/Type\s*/Page[^s]", data))
    print(f"report/report.pdf: {pages} pages (limit {a.max_pages})")
    sys.exit(0 if 0 < pages <= a.max_pages else 1)


if __name__ == "__main__":
    main()
