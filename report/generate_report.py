"""
generate_report.py
--------------------
The full "summary.json -> Markdown -> PDF" pipeline in one command.

Usage:
    python3 -m report.generate_report --outdir ./output --eval-outdir ./output/evaluation

Produces:
    report/REPORT.md   - human-readable markdown (also useful for git diffs)
    report/REPORT.docx - polished Word version (pandoc markdown->docx)
    report/REPORT.pdf  - final PDF deliverable (LibreOffice headless conversion)
"""

import argparse
import os
import subprocess
import sys

from .gather_data import gather
from .render_markdown import render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="./output")
    parser.add_argument("--eval-outdir", default="./output/evaluation")
    parser.add_argument("--report-dir", default="./report")
    args = parser.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)

    print("[1/3] Gathering live data from evaluation results + database...")
    data = gather(outdir=args.outdir, eval_outdir=args.eval_outdir)
    print(f"       -> {data['dataset']['total_sessions']:,} sessions, "
          f"{data['total_alerts_in_queue']:,} alerts, "
          f"precision={data['detection_metrics']['precision']:.3f}")

    print("[2/3] Rendering Markdown report...")
    markdown = render(data)
    md_path = os.path.join(args.report_dir, "REPORT.md")
    with open(md_path, "w") as f:
        f.write(markdown)
    print(f"       -> wrote {md_path} ({len(markdown):,} chars)")

    print("[3/3] Converting Markdown -> DOCX -> PDF...")
    docx_path = os.path.join(args.report_dir, "REPORT.docx")
    pdf_path = os.path.join(args.report_dir, "REPORT.pdf")

    # Markdown -> DOCX via pandoc. Run from report_dir so relative image
    # paths in the markdown (../output/evaluation/*.png) resolve correctly
    # regardless of where this script itself is invoked from.
    result = subprocess.run(
        ["pandoc", "REPORT.md", "-o", "REPORT.docx", "--resource-path=.."],
        cwd=args.report_dir, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("❌ pandoc failed:")
        print(result.stderr)
        sys.exit(1)
    print(f"       -> wrote {docx_path}")

    # DOCX -> PDF via headless LibreOffice
    result = subprocess.run(
        ["soffice", "--headless", "--convert-to", "pdf", "REPORT.docx"],
        cwd=args.report_dir, capture_output=True, text=True,
    )
    if result.returncode != 0 or not os.path.exists(pdf_path):
        print("❌ soffice PDF conversion failed:")
        print(result.stdout, result.stderr)
        sys.exit(1)
    print(f"       -> wrote {pdf_path}")

    print(f"\nDone. Final report: {pdf_path}")


if __name__ == "__main__":
    main()
