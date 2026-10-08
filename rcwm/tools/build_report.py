#!/usr/bin/env python3
"""Build the master PDF and self-contained HTML using installed Pandoc/LaTeX.

Does not install tools, fonts or models. Run from any directory.
"""
from pathlib import Path
import subprocess,shutil
R=Path(__file__).resolve().parents[1]
if not shutil.which('pandoc') or not shutil.which('xelatex'):
    raise SystemExit('Install Pandoc and a XeLaTeX distribution separately; no automatic install is performed.')
parts=sorted((R/'report').glob('[0-9][0-9]_*.md'))
master=R/'report/Rineya_RCWM_Research_Plan.md'
master.write_text('\n\n'.join(('\\clearpage\n\n' if p.name.startswith(('10_', '99_')) else '') + p.read_text() for p in parts))
base=['pandoc',str(master),'--standalone','--number-sections','--toc','--toc-depth=1',
      '-V','documentclass=article','-V','fontsize=11pt','-V','papersize=a4',
      '-V','geometry:margin=22mm','-V','mainfont=Liberation Serif',
      '-V','sansfont=Lato','-V','monofont=DejaVu Sans Mono',
      '--include-in-header='+str(R/'tools/report_style.tex'),
      '--include-before-body='+str(R/'tools/report_cover.tex')]
subprocess.run(base+['--pdf-engine=xelatex','-o',str(R/'report/Rineya_RCWM_Research_Plan.pdf')],check=True)
# Standalone editable LaTeX build source (depends only on installed packages/fonts).
subprocess.run(base+['-t','latex','-o',str(R/'report/Rineya_RCWM_Research_Plan.tex')],check=True)
subprocess.run(['pandoc',str(master),'--standalone','--number-sections','--toc','--toc-depth=1',
                '--mathml','--embed-resources','--metadata','title=Rineya Consequence World Model — Research Plan',
                '--css='+str(R/'tools/offline_style.css'),'-o',str(R/'report/Rineya_RCWM_Research_Plan.html')],check=True)
