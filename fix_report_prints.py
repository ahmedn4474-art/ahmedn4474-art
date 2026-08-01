"""Fix Financial and verify report prints in all projects"""
import os, re

base = r'D:\download\protfolio\portfolio_projects'

for proj in sorted(os.listdir(base)):
    path = os.path.join(base, proj, 'project.py')
    if not os.path.exists(path):
        continue
    
    with open(path, encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    new_lines = []
    changes = 0
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        s = line.strip()
        
        # Match f.write(some_var) or f.write(some_var.strip()...) 
        m = re.match(r'f\.write\((\w+)(?:\.\w+\([^)]*\))?(?:\s*\+\s*[^)]*)?\)', s)
        if m:
            var_name = m.group(1)
            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(f'{indent}print({var_name})')
            changes += 1
    
    if changes:
        content = '\n'.join(new_lines)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f'{proj}: {changes} print() added')

print('\nDone')
