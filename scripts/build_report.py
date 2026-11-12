#!/usr/bin/env python3
"""Convert docs/report.md to LaTeX and compile it with XeLaTeX (no Python dependencies)."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'


def escape(text):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}', '≥': r'$\geq$', '−': '-', 'ε': r'$\varepsilon$'}
    return ''.join(replacements.get(char, char) for char in text)


def inline(text):
    parts = re.split(r'(`[^`]+`|\*\*[^*]+\*\*)', text)
    result = []
    for part in parts:
        if part.startswith('`'):
            # URLs allow line breaking in long artifact names and paths.
            result.append(r'\path{' + part[1:-1] + '}')
        elif part.startswith('**'):
            result.append(r'\textbf{' + escape(part[2:-2]) + '}')
        else:
            result.append(escape(part))
    return ''.join(result)


def render(source):
    lines = source.splitlines()
    output = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('# '):
            output.append(r'\maketitle')
        elif line.startswith('### '):
            title = re.sub(r'^\d+\.\d+\s+', '', line[4:])
            output.append(r'\subsection{' + inline(title) + '}')
        elif line.startswith('## '):
            title = re.sub(r'^\d+\.\s+', '', line[3:])
            command = 'section*' if title == 'Abstract' else 'section'
            output.append('\\' + command + '{' + inline(title) + '}')
        elif line.startswith('```') or line.startswith('    '):
            fenced = line.startswith('```')
            block = [] if fenced else [line[4:]]
            i += 1
            while i < len(lines) and (not lines[i].startswith('```') if fenced else lines[i].startswith('    ')):
                block.append(lines[i] if fenced else lines[i][4:])
                i += 1
            output.extend([r'\begin{lstlisting}[basicstyle=\ttfamily\small,breaklines=true,columns=fullflexible]', '\n'.join(block).replace('ε', 'epsilon').replace('−', '-').replace('≥', '>='), r'\end{lstlisting}'])
            if not fenced:
                continue
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [v.strip() for v in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', v) for v in cells):
                    rows.append(cells)
                i += 1
            count = len(rows[0])
            output.extend([r'\begin{center}\footnotesize\setlength{\tabcolsep}{3pt}', r'\begin{tabularx}{\textwidth}{' + 'X' * count + '}', r'\toprule'])
            for index, row in enumerate(rows):
                output.append(' & '.join(inline(v) for v in row) + r' \\')
                if index == 0:
                    output.append(r'\midrule')
            output.extend([r'\bottomrule', r'\end{tabularx}\end{center}'])
            continue
        elif line.startswith('!['):
            match = re.fullmatch(r'!\[([^]]*)\]\(([^)]+)\)', line)
            if not match:
                raise ValueError('Unsupported image syntax: ' + line)
            path = (DOCS / match[2]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            # Bundle report images so the LaTeX source is portable.
            destination = DOCS / 'report_assets' / (path.parent.name + '_' + path.name)
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, destination)
            output.extend([r'\begin{center}', r'\includegraphics[width=\linewidth]{report_assets/' + destination.name + '}', r'\end{center}'])
        elif re.match(r'^(\d+\. |\- )', line):
            numbered = not line.startswith('- ')
            env = 'enumerate' if numbered else 'itemize'
            output.append(r'\begin{' + env + '}')
            while i < len(lines) and re.match(r'^\d+\. ' if numbered else r'^- ', lines[i]):
                output.append(r'\item ' + inline(re.sub(r'^(\d+\. |\- )', '', lines[i])))
                i += 1
            output.append(r'\end{' + env + '}')
            continue
        elif line.startswith('*') and line.endswith('*') and not line.startswith('**'):
            output.append(r'\noindent\textit{' + inline(line[1:-1]) + r'}\par')
        else:
            output.append(inline(line))
        i += 1
    return '\n'.join(output)


def main():
    engine = shutil.which('xelatex')
    if not engine:
        raise SystemExit('XeLaTeX is required. Install a TeX distribution such as MacTeX.')
    preamble = r'''\documentclass[11pt]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\usepackage{graphicx,booktabs,tabularx,listings}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\setlength{\emergencystretch}{3em}
\title{FocusDepth: Confidence-Guided Fusion of Focus-Based and Neural Depth}
\author{Godwin Kangor}
\date{Working report --- November 2, 2026}
\begin{document}
'''
    tex = DOCS / 'report.tex'
    tex.write_text(preamble + render((DOCS / 'report.md').read_text()) + '\n\\end{document}\n')
    with tempfile.TemporaryDirectory(prefix='focusdepth-report-') as temporary:
        for _ in range(2):
            result = subprocess.run([engine, '-no-shell-escape', '-interaction=nonstopmode', '-halt-on-error', '-output-directory=' + temporary, tex.name], cwd=DOCS, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if result.returncode:
                raise SystemExit(result.stdout)
        shutil.copyfile(Path(temporary) / 'report.pdf', DOCS / 'report.pdf')
        log = (Path(temporary) / 'report.log').read_text()
        for line in log.splitlines():
            if 'Overfull' in line or 'Missing character' in line:
                print(line)
    print('Created docs/report.tex and docs/report.pdf from docs/report.md')


if __name__ == '__main__':
    main()
