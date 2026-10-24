#!/usr/bin/env python
"""docs/REPORT.md -> report/report.pdf (pandoc body + minimal XeLaTeX wrapper with a Unicode font), then the page count.

    python scripts/build_report.py [--max-pages 4]
Exit 1 if the PDF is longer than --max-pages.  Falls back to pandoc's default template with pdflatex if xelatex
is missing (then Unicode symbols in tables must be avoided).
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
\usepackage[margin=1.7cm]{geometry}
\usepackage{fontspec}
\setmainfont{DejaVu Serif}[Scale=0.92]
\setmonofont{DejaVu Sans Mono}[Scale=0.85]
\usepackage{amsmath,amssymb}
\usepackage{booktabs,longtable,array,calc}
\usepackage{graphicx}
\setkeys{Gin}{keepaspectratio}
\usepackage{hyperref}
\usepackage{float}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\setlength{\parskip}{3pt}
\setlength{\parindent}{0pt}
\setcounter{secnumdepth}{0}
\title{%(title)s}
\author{%(subtitle)s}
\date{%(date)s}
\begin{document}
\maketitle
\vspace{-1.5em}
%(body)s
\end{document}
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=4)
    a = ap.parse_args(argv)
    src = ROOT / "docs/REPORT.md"
    out = ROOT / "report"
    out.mkdir(exist_ok=True)
    text = src.read_text(encoding="utf-8")
    meta = dict(re.findall(r'^(title|subtitle|date):\s*"?(.*?)"?\s*$', text, flags=re.M))
    body = subprocess.run(["pandoc", str(src), "-t", "latex", "--from", "markdown"], capture_output=True, text=True, check=True).stdout
    tex = WRAPPER % {"title": meta.get("title", "Report"), "subtitle": meta.get("subtitle", ""), "date": meta.get("date", ""), "body": body}
    (out / "report.tex").write_text(tex, encoding="utf-8")
    engine = "xelatex" if shutil.which("xelatex") else "pdflatex"
    for _ in range(2):
        r = subprocess.run([engine, "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd=out, capture_output=True, text=True)
    if not (out / "report.pdf").exists():
        print(r.stdout[-3000:])
        sys.exit("LaTeX failed")
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
