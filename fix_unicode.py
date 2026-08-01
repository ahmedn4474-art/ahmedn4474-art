"""Fix all non-ASCII characters in project files (except Arabic)"""
import os, glob

base = 'D:/download/protfolio/portfolio_projects'
files = glob.glob(os.path.join(base, '*/project.py')) + [os.path.join(base, '_common.py')]

replacements = {
    '\u2014': '--',  # em dash
    '\u2013': '-',   # en dash
    '\u2500': '-',   # box horizontal
    '\u2550': '=',   # double box horizontal
    '\u2514': '+',   # box corner
    '\u251c': '+',   # box tee
    '\u2502': '|',   # box vertical
    '\u2022': '*',   # bullet
    '\u2018': "'",   # left smart quote
    '\u2019': "'",   # right smart quote
    '\u201c': '"',   # left double quote
    '\u201d': '"',   # right double quote
    '\u00b2': '^2',  # superscript 2
    '\u00d7': 'x',   # multiplication
    '\u00e9': 'e',   # e-acute
    '\u03c7': 'chi', # chi
    '\u2264': '<=',  # less-than-or-equal
    '\u2265': '>=',  # greater-than-or-equal
    '\u00b1': '+/-', # plus-minus
    '\u2905': '->',  # rightwards arrow with plus
    '\u2192': '->',  # rightwards arrow
    '\u27f6': '->',  # long rightwards arrow
    '\u00b7': '*',   # middle dot
}

for f in files:
    with open(f, 'r', encoding='utf-8') as fh:
        content = fh.read()
    new_chars = []
    for ch in content:
        if ord(ch) > 127:
            if 0x600 <= ord(ch) <= 0x6ff or 0x750 <= ord(ch) <= 0x77f:
                new_chars.append(ch)
            elif ch in replacements:
                new_chars.append(replacements[ch])
            else:
                print(f"  UNKNOWN: {f} U+{ord(ch):04X} = {repr(ch)}")
                new_chars.append(f'[{ord(ch):04X}]')
        else:
            new_chars.append(ch)
    content = ''.join(new_chars)
    with open(f, 'w', encoding='utf-8') as fh:
        fh.write(content)
    print(f'Fixed: {os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}')

print('Done')
