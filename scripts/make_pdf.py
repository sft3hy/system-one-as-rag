"""Render WEEKLY-REPORT.md -> shareable PDF.

Pipeline: markdown -> styled HTML (images inlined as base64 so the PDF is
self-contained) -> headless Chrome print-to-pdf. No pandoc on this box;
Chrome is. Re-run after editing the markdown.
"""
import base64
import pathlib
import re
import subprocess

import markdown

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "WEEKLY-REPORT.md"
HTML = ROOT / "WEEKLY-REPORT.pdf.html"
PDF = ROOT / "WEEKLY-REPORT.pdf"

text = SRC.read_text()

# inline every results/... png as base64 before conversion
def inline(m):
    rel, = m.groups()
    img = (ROOT / rel)
    if not img.exists():
        raise SystemExit(f"referenced image missing: {rel}")
    b64 = base64.b64encode(img.read_bytes()).decode()
    return f"![{rel}](data:image/png;base64,{b64})"

text = re.sub(r"!\[[^\]]*\]\((results/[^)]+\.png)\)", inline, text)

body = markdown.markdown(text, extensions=["tables", "sane_lists", "fenced_code"])

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Weekly Activity Report — AI Answer Safety Gate</title>
<style>
  @page {{ size: A4; margin: 18mm 16mm; }}
  body {{ font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif;
         font-size: 10.5pt; line-height: 1.55; color: #1a1a1a; max-width: 100%; }}
  h1 {{ font-size: 19pt; border-bottom: 3px solid #d62728; padding-bottom: 6px; margin-top: 0; }}
  h2 {{ font-size: 14pt; margin-top: 22px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  h3 {{ font-size: 11.5pt; margin-top: 16px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.3pt; }}
  th, td {{ border: 1px solid #ccc; padding: 5px 8px; text-align: left; vertical-align: top; }}
  th {{ background: #f2f2f2; }}
  tr:nth-child(even) td {{ background: #fafafa; }}
  blockquote {{ border-left: 4px solid #9467bd; margin: 10px 0; padding: 6px 14px;
               background: #f7f5fb; border-radius: 0 6px 6px 0; }}
  img {{ max-width: 100%; height: auto; display: block; margin: 12px auto;
         border: 1px solid #e5e5e5; border-radius: 6px; }}
  code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }}
  hr {{ border: none; border-top: 1px solid #ddd; margin: 18px 0; }}
  /* keep figures with their headings; break before major sections */
  h2 {{ break-after: avoid; }}
  table, blockquote, img {{ break-inside: avoid; }}
  .footer {{ color: #777; font-size: 8.5pt; }}
</style></head><body>
{body}
<p class="footer">Generated {__import__('datetime').date.today().isoformat()} — full data:
results/full-benchmark-v1 (2,780 conditions/arm, zero errors).</p>
</body></html>"""

HTML.write_text(html)

chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
r = subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                   f"--print-to-pdf={PDF}", str(HTML)], capture_output=True, text=True, timeout=120)
if not PDF.exists():
    raise SystemExit(f"chrome failed: {r.stderr[-500:]}")
print(f"wrote {PDF} ({PDF.stat().st_size/1024:.0f} KB)")
