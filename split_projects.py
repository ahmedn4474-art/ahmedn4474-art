import os, subprocess, shutil, stat

def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

projects = [
    {
        "folder": r"D:\download\protfolio\Project1_HR_Attrition",
        "repo": "HR-Employee-Attrition-Analysis"
    },
    {
        "folder": r"D:\download\protfolio\Project2_Twitter_Sentiment",
        "repo": "Twitter-Sentiment-Classification"
    },
    {
        "folder": r"D:\download\protfolio\Project3_Audit_Risk",
        "repo": "Audit-Risk-Anomaly-Detection"
    },
    {
        "folder": r"D:\download\protfolio\Project4_Financial_Accounting",
        "repo": "Financial-Accounting-Analytics"
    },
    {
        "folder": r"D:\download\protfolio\Project5_Corporate_Bankruptcy",
        "repo": "Corporate-Bankruptcy-Prediction"
    }
]

for p in projects:
    folder = p["folder"]
    repo_name = p["repo"]
    git_dir = os.path.join(folder, ".git")
    if os.path.exists(git_dir):
        shutil.rmtree(git_dir, onerror=remove_readonly)
    
    print(f"Initializing independent Git repository for {repo_name}...")
    subprocess.run("git init", shell=True, cwd=folder)
    subprocess.run("git branch -M main", shell=True, cwd=folder)
    url = f"https://github.com/ahmedn4474-art/{repo_name}.git"
    subprocess.run(f"git remote add origin {url}", shell=True, cwd=folder)
    
    # Create gitignore if not present
    gitignore = os.path.join(folder, ".gitignore")
    with open(gitignore, "w", encoding="utf-8") as f:
        f.write("*.zip\n*.csv\n.ipynb_checkpoints/\n__pycache__/\n")
    
    subprocess.run("git add .", shell=True, cwd=folder)
    subprocess.run(f'git commit -m "Initial release of {repo_name}"', shell=True, cwd=folder)

print("\nALL 5 PROJECTS PREPARED AS SEPARATE INDEPENDENT GIT REPOSITORIES!")
