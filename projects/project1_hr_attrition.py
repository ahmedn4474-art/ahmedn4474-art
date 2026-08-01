import pandas as pd, numpy as np
from scipy import stats
from scipy.stats import beta as beta_dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

OUT = "D:\\download\\protfolio\\projects\\output\\project1_HR"
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT,"report.txt"),'w',encoding='utf-8')
def w(t=""): log.write(t+"\n"); print(t.encode("utf-8","replace").decode("utf-8","replace"))

hr = pd.read_csv("D:\\download\\protfolio\\archive\\WA_Fn-UseC_-HR-Employee-Attrition.csv")
w("="*80 + "\n  PROJECT 1: HR EMPLOYEE ATTRITION ANALYSIS\n  ØªØ­Ù„ÙŠÙ„ Ø§Ø³ØªÙ‚Ø§Ù„Ø§Øª Ø§Ù„Ù…ÙˆØ¸ÙÙŠÙ†\n" + "="*80)

# ----- 1. Frequency Statistics -----
w("\n" + "-"*60 + "\n  1. FREQUENCY STATISTICS / Ø§Ù„Ø¥Ø­ØµØ§Ø¡ Ø§Ù„ØªÙƒØ±Ø§Ø±ÙŠ\n" + "-"*60)
for col in ['Attrition','Gender','Department','JobRole','MaritalStatus','OverTime','BusinessTravel']:
    f = hr[col].value_counts(); p = hr[col].value_counts(normalize=True).mul(100).round(1)
    w(f"\n{col}:"); [w(f"  {k:30s}: {f[k]:5d} ({p[k]:.1f}%)") for k in f.index]

hr['AgeG'] = pd.cut(hr['Age'],[18,25,35,45,55,65],labels=['18-25','26-35','36-45','46-55','56-65'])
hr['IncG'] = pd.cut(hr['MonthlyIncome'],[0,3000,6000,10000,15000,20000],labels=['<3K','3K-6K','6K-10K','10K-15K','15K+'])
hr['TenG'] = pd.cut(hr['YearsAtCompany'],[0,2,5,10,20,40],labels=['<2yr','2-5yr','5-10yr','10-20yr','20+yr'])
w("\nBinned Variables:"); [w(f"  Age {k}: {v}") for k,v in hr['AgeG'].value_counts().sort_index().items()]
w(""); [w(f"  Income {k}: {v}") for k,v in hr['IncG'].value_counts().sort_index().items()]
w(""); [w(f"  Tenure {k}: {v}") for k,v in hr['TenG'].value_counts().sort_index().items()]

w("\nDescriptive Stats (Numeric):")
w(hr[['Age','MonthlyIncome','YearsAtCompany','DistanceFromHome','PercentSalaryHike','TotalWorkingYears']].describe().round(2).to_string())

# ----- 2. A/B Testing -----
w("\n" + "-"*60 + "\n  2. A/B HYPOTHESIS TESTING / Ø§Ø®ØªØ¨Ø§Ø± Ø§Ù„ÙØ±Ø¶ÙŠØ§Øª\n" + "-"*60)

tests = [
    ("OverTime vs Attrition", pd.crosstab(hr['OverTime'],hr['Attrition'])),
    ("Gender vs Attrition", pd.crosstab(hr['Gender'],hr['Attrition'])),
    ("Department vs Attrition", pd.crosstab(hr['Department'],hr['Attrition'])),
    ("BusinessTravel vs Attrition", pd.crosstab(hr['BusinessTravel'],hr['Attrition'])),
    ("MaritalStatus vs Attrition", pd.crosstab(hr['MaritalStatus'],hr['Attrition'])),
]
for name, ct in tests:
    chi2, p = stats.chi2_contingency(ct)[:2]
    sig = "*** SIGNIFICANT" if p<0.001 else "** SIGNIFICANT" if p<0.01 else "* SIGNIFICANT" if p<0.05 else "ns"
    w(f"\n{name}:\n{ct}\n  chi2={chi2:.2f}, p={p:.6f} [{sig}]")

ttests = [
    ("MonthlyIncome", hr[hr.Attrition=='Yes']['MonthlyIncome'], hr[hr.Attrition=='No']['MonthlyIncome']),
    ("YearsAtCompany", hr[hr.Attrition=='Yes']['YearsAtCompany'], hr[hr.Attrition=='No']['YearsAtCompany']),
    ("Age", hr[hr.Attrition=='Yes']['Age'], hr[hr.Attrition=='No']['Age']),
    ("DistanceFromHome", hr[hr.Attrition=='Yes']['DistanceFromHome'], hr[hr.Attrition=='No']['DistanceFromHome']),
]
for name, g1, g2 in ttests:
    t, p = stats.ttest_ind(g1, g2)
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    w(f"\nT-Test: {name} by Attrition\n  Yes: {g1.mean():.2f} | No: {g2.mean():.2f}\n  t={t:.3f}, p={p:.6f} [{sig}]")

# ANOVA
for col in ['JobSatisfaction','WorkLifeBalance','EnvironmentSatisfaction','JobInvolvement']:
    f, p = stats.f_oneway(hr[hr.Attrition=='Yes'][col], hr[hr.Attrition=='No'][col])
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    w(f"\nANOVA: {col} by Attrition\n  Yes: {hr[hr.Attrition=='Yes'][col].mean():.2f} | No: {hr[hr.Attrition=='No'][col].mean():.2f}\n  F={f:.2f}, p={p:.6f} [{sig}]")

# ----- 3. Bayesian Analysis -----
w("\n" + "-"*60 + "\n  3. BAYESIAN ANALYSIS / Ø§Ù„ØªØ­Ù„ÙŠÙ„ Ø§Ù„Ø¨ÙŠØ²ÙŠ\n" + "-"*60)
a_prior,b_prior = 1,1

k_all = int((hr.Attrition=='Yes').sum()); n_all = len(hr)
a_all,b_all = a_prior+k_all, b_prior+(n_all-k_all)
w(f"\nOverall Attrition: Beta({a_all},{b_all})")
w(f"  Mean: {a_all/(a_all+b_all)*100:.2f}%")
w(f"  95% CI: [{beta_dist.ppf(0.025,a_all,b_all)*100:.2f}%, {beta_dist.ppf(0.975,a_all,b_all)*100:.2f}%]")

w("\nBayesian A/B: OverTime")
for ot_val, label in [('Yes','OT=Yes'),('No','OT=No')]:
    sub = hr[hr.OverTime==ot_val]; k = int((sub.Attrition=='Yes').sum()); n = len(sub)
    a,b = a_prior+k, b_prior+(n-k)
    w(f"  {label}: Beta({a},{b}) mean={a/(a+b)*100:.2f}% [{beta_dist.ppf(0.025,a,b)*100:.2f}-{beta_dist.ppf(0.975,a,b)*100:.2f}]")

ky_ot = int((hr[(hr.OverTime=='Yes')&(hr.Attrition=='Yes')].shape[0]))
ny_ot = int((hr.OverTime=='Yes').sum())
kn_ot = int((hr[(hr.OverTime=='No')&(hr.Attrition=='Yes')].shape[0]))
nn_ot = int((hr.OverTime=='No').sum())
sy = beta_dist.rvs(a_prior+ky_ot, b_prior+(ny_ot-ky_ot), 200000)
sn = beta_dist.rvs(a_prior+kn_ot, b_prior+(nn_ot-kn_ot), 200000)
w(f"  P(OT=Yes > OT=No) = {(sy>sn).mean()*100:.2f}%")
w(f"  Relative Risk posterior: mean={(sy/sn).mean():.2f} | 95% CI: [{np.percentile(sy/sn,2.5):.2f}, {np.percentile(sy/sn,97.5):.2f}]")

w("\nBayesian by Department:")
depts = []
for d in hr.Department.unique():
    sub = hr[hr.Department==d]; k = int((sub.Attrition=='Yes').sum()); n = len(sub)
    a,b = a_prior+k, b_prior+(n-k); m = a/(a+b)
    lo,hi = beta_dist.ppf(0.025,a,b), beta_dist.ppf(0.975,a,b)
    depts.append((m,d,lo,hi))
    w(f"  {d:25s}: {m*100:.1f}% [{lo*100:.1f}%-{hi*100:.1f}%]")
depts.sort(reverse=True); w("  Risk Ranking:"); [w(f"    #{i} {d:25s} {m*100:.1f}%") for i,(m,d,_,_) in enumerate(depts,1)]

w("\nBayesian by MaritalStatus:")
for ms in hr.MaritalStatus.unique():
    sub = hr[hr.MaritalStatus==ms]; k = int((sub.Attrition=='Yes').sum()); n = len(sub)
    a,b = a_prior+k, b_prior+(n-k); lo,hi = beta_dist.ppf(0.025,a,b), beta_dist.ppf(0.975,a,b)
    w(f"  {ms:15s}: {k:3d}/{n:4d} -> {a/(a+b)*100:.1f}% [{lo*100:.1f}%-{hi*100:.1f}%]")

w("\nBayesian by BusinessTravel:")
for bt in hr.BusinessTravel.unique():
    sub = hr[hr.BusinessTravel==bt]; k = int((sub.Attrition=='Yes').sum()); n = len(sub)
    a,b = a_prior+k, b_prior+(n-k); lo,hi = beta_dist.ppf(0.025,a,b), beta_dist.ppf(0.975,a,b)
    w(f"  {bt:25s}: {k:3d}/{n:4d} -> {a/(a+b)*100:.1f}% [{lo*100:.1f}%-{hi*100:.1f}%]")

# ----- 4. Visualizations -----
w("\n" + "-"*60 + "\n  4. VISUALIZATIONS / Ø§Ù„Ø±Ø³ÙˆÙ… Ø§Ù„Ø¨ÙŠØ§Ù†ÙŠØ©\n" + "-"*60)
fig, axes = plt.subplots(2,3,figsize=(18,12))
fig.suptitle('Project 1: HR Attrition - Professional Analysis', fontsize=16, fontweight='bold')

hr['Attrition'].value_counts().plot(kind='bar',ax=axes[0,0],color=['#2ecc71','#e74c3c'],edgecolor='k')
axes[0,0].set_title('Attrition Distribution',fontweight='bold')

pd.crosstab(hr['OverTime'],hr['Attrition'],normalize='index').plot(kind='bar',ax=axes[0,1],stacked=True,color=['#2ecc71','#e74c3c'],edgecolor='k')
axes[0,1].set_title('Attrition by OverTime',fontweight='bold')

hr.groupby('Department')['Attrition'].apply(lambda x: (x=='Yes').mean()*100).sort_values().plot(kind='barh',ax=axes[0,2],color='#3498db',edgecolor='k')
axes[0,2].set_title('Attrition % by Dept',fontweight='bold'); axes[0,2].set_xlabel('%')

hr.boxplot(column='MonthlyIncome',by='Attrition',ax=axes[1,0]); axes[1,0].set_title('Income by Attrition',fontweight='bold')
hr.boxplot(column='YearsAtCompany',by='Attrition',ax=axes[1,1]); axes[1,1].set_title('Tenure by Attrition',fontweight='bold')

x=np.linspace(0,0.5,500)
axes[1,2].plot(x,beta_dist.pdf(x,a_prior+ky_ot,b_prior+(ny_ot-ky_ot)),'r-',lw=2.5,label='OT=Yes')
axes[1,2].plot(x,beta_dist.pdf(x,a_prior+kn_ot,b_prior+(nn_ot-kn_ot)),'g-',lw=2.5,label='OT=No')
axes[1,2].fill_between(x,beta_dist.pdf(x,a_prior+ky_ot,b_prior+(ny_ot-ky_ot)),alpha=0.1,color='red')
axes[1,2].fill_between(x,beta_dist.pdf(x,a_prior+kn_ot,b_prior+(nn_ot-kn_ot)),alpha=0.1,color='green')
axes[1,2].set_title('Bayesian Posteriors',fontweight='bold'); axes[1,2].legend()
plt.tight_layout()
fig.savefig(f"{OUT}\\HR_Project_Analysis.png",dpi=150,bbox_inches='tight')
w("  [OK] HR_Project_Analysis.png")

# Additional chart: Correlation heatmap
fig2, ax2 = plt.subplots(figsize=(10,8))
num_cols = ['Age','DailyRate','DistanceFromHome','MonthlyIncome','NumCompaniesWorked','PercentSalaryHike','TotalWorkingYears','YearsAtCompany','JobSatisfaction','WorkLifeBalance']
corr = hr[num_cols].corr()
im = ax2.imshow(corr, cmap='RdYlBu_r', vmin=-1, vmax=1)
ax2.set_xticks(range(len(num_cols))); ax2.set_yticks(range(len(num_cols)))
ax2.set_xticklabels(num_cols,rotation=45,ha='right'); ax2.set_yticklabels(num_cols)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax2.text(j,i,f"{corr.iloc[i,j]:.2f}",ha='center',va='center',fontsize=8)
ax2.set_title('Correlation Matrix - HR Attrition',fontweight='bold')
fig2.colorbar(im,ax=ax2)
fig2.savefig(f"{OUT}\\HR_Correlation.png",dpi=150,bbox_inches='tight')
w("  [OK] HR_Correlation.png")

log.close()
print(f"\nPROJECT 1 COMPLETE -> {OUT}\\report.txt")

