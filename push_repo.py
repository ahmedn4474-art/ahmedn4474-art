import subprocess, sys

def run(cmd):
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=r"D:\download\protfolio")
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
    return res.returncode

print("Checking git status...")
run("git status")

print("Pushing to GitHub...")
code = run("git push -u origin main --force")
if code == 0:
    print("\nSUCCESSFULLY PUSHED ALL 5 PROJECTS TO GITHUB!")
else:
    print("\nPush returned exit code:", code)
