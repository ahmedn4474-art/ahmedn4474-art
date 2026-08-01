import shutil, os, stat, subprocess

def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

p = r"D:\download\protfolio"
g = os.path.join(p, ".git")

if os.path.exists(g):
    print("Removing old git history containing heavy 227MB CSV...")
    shutil.rmtree(g, onerror=remove_readonly)

print("Initializing fresh clean git repository...")
subprocess.run("git init", shell=True, cwd=p)
subprocess.run("git branch -M main", shell=True, cwd=p)
subprocess.run("git remote add origin https://github.com/ahmedn4474-art/ahmedn4474-art.git", shell=True, cwd=p)
subprocess.run("git add .", shell=True, cwd=p)
subprocess.run('git commit -m "Clean portfolio release: 5 data projects and elite profile README"', shell=True, cwd=p)

print("\nDONE! Ready to push cleanly!")
