import subprocess, sys

print("Executing direct Git push to GitHub repository ahmedn4474-art/ahmedn4474-art...")
res = subprocess.run(["git", "push", "origin", "main", "--force"], cwd=r"D:\download\protfolio", capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
print("Exit Code:", res.returncode)

if res.returncode == 0:
    print("SUCCESS: ALL 5 PROJECTS AND ANIMATED README PUSHED TO GITHUB!")
else:
    print("PUSH FAILED WITH CODE:", res.returncode)
