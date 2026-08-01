"""Fix encoding corruption in all project.py files"""
import glob, os, re

bad_dir = r'D:\download\protfolio\portfolio_projects'

# Translation table for common corruption patterns
# UTF-8 bytes misinterpreted as cp1252
corruptions = {
    'â€”': '\u2014',   # — (em-dash)
    'â”€': '\u2500',   # ─ (box drawing horizontal)
    'â™¦': '\u2666',   # ♦
    'Ã©': '\u00e9',    # é
    'Ã¼': '\u00fc',    # ü
    'Å“': '\u0153',    # œ (ligature)
}

for fn in glob.glob(os.path.join(bad_dir, '*', 'project.py')):
    with open(fn, encoding='utf-8-sig') as f:
        content = f.read()
    
    original = content
    for bad, good in corruptions.items():
        content = content.replace(bad, good)
    # Also try to fix generic corrupted em-dash: \xe2\x80\x94 read as other encoding
    # The ── pattern: the corrupted bytes for ── (U+2500 U+2500) in cp1252
    # When written as UTF-8 then read as Latin-1: â”€â”€
    # The em-dash — (U+2014) in UTF-8 is 0xE2 0x80 0x94
    # In cp1252 that displays as â€" or â€”
    
    # Check if content actually changed
    if content != original:
        with open(fn, 'w', encoding='utf-8') as f:
            f.write(content)
        name = os.path.basename(os.path.dirname(fn))
        diff = sum(1 for a, b in zip(content, original) if a != b)
        print(f'Fixed {name}: {diff} chars changed')
    else:
        name = os.path.basename(os.path.dirname(fn))
        print(f'OK {name}')

print('Done')
