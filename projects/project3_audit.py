import pandas as pd, numpy as np
from scipy import stats
from scipy.stats import beta as beta_dist
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings, subprocess, json
warnings.filterwarnings('ignore')

OUT = "D:\\download\\protfolio\\projects\\output\\project3_Audit"
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT,"report.txt"),'w',encoding='utf-8')
def w(t=""): log.write(t+"\n"); print(t)

w("="*80 + "\n  PROJECT 3: AUDIT & SECURITY ANALYSIS\n" + "="*80)

# Read XLSX via PowerShell CSV export
w("\nLoading audit dataset...")
ps_script = '''
$excel = New-Object -ComObject Excel.Application; $excel.Visible = $false
$wb = $excel.Workbooks.Open("D:\\download\\protfolio\\archive (2)\\full_audit_dataset_with_security_operational.xlsx")
$ws = $wb.Sheets.Item(1)
$csvPath = [System.IO.Path]::GetTempFileName() + ".csv"
$ws.SaveAs($csvPath, 6)
$wb.Close($false); $excel.Quit()
[Runtime.Interopservices.Marshal]::ReleaseComObject($excel) | Out-Null
Write-Host $csvPath
'''
result = subprocess.run(['powershell','-NoProfile','-Command',ps_script], capture_output=True, text=True)
csv_path = result.stdout.strip().split('\n')[-1].strip()
w(f"  Temp CSV: {csv_path}")
audit_data = pd.read_csv(csv_path)
os.remove(csv_path)
w(f"  Loaded: {len(audit_data)} audits x {len(audit_data.columns)} cols")

# Fix column names
audit_data.columns = ['AuditID','AuditType','DataValue','Timestamp','AuditScore','Variance','Duration','RiskLevel','AuditStatus','ErrorRate','CompletionPct','AuditCost','ReviewCount','RiskFactor',
'PaymentProcessingSecurity','PCIDSS','EncryptionStandard','TokenizationUsed','PaymentGatewaySecurity','WebsiteSecurity','VulnerabilityTesting','WAFInPlace','AccessControls','MalwareScanning','PatchManagement','PasswordPolicies','AccountTakeoverPrevention','SessionManagement','StockoutsRisk','OverstockingRisk','InventoryDataAccuracy','OrderAccuracy','ShippingIssues','ShippingCostAccuracy','ReturnsRefundsProcess']

w("\n" + "-"*60 + "\n  1. FREQUENCY STATISTICS\n" + "-"*60)
for col in ['AuditType','RiskLevel','AuditStatus']:
    f = audit_data[col].value_counts(); p = audit_data[col].value_counts(normalize=True).mul(100).round(1)
    w(f"\n{col}:"); [w(f"  {k:20s}: {f[k]:4d} ({p[k]:.1f}%)") for k in f.index]

w("\nDescriptive Stats:")
w(audit_data[['AuditScore','Variance','Duration','ErrorRate','CompletionPct','AuditCost','RiskFactor']].describe().round(2).to_string())

w("\n" + "-"*60 + "\n  2. A/B TESTING\n" + "-"*60)
tests = [
    ("AuditType vs RiskLevel", pd.crosstab(audit_data['AuditType'],audit_data['RiskLevel'])),
    ("AuditStatus vs RiskLevel", pd.crosstab(audit_data['AuditStatus'],audit_data['RiskLevel'])),
]
for name, ct in tests:
    chi2,p = stats.chi2_contingency(ct)[:2]
    w(f"\n{name}:\n{ct}\n  chi2={chi2:.2f}, p={p:.6f} {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'}")

w("\nANOVA: AuditScore by RiskLevel")
for rl in ['Low','Medium','High']:
    w(f"  {rl}: mean={audit_data[audit_data.RiskLevel==rl]['AuditScore'].mean():.1f}")
f,p = stats.f_oneway(*[audit_data[audit_data.RiskLevel==rl]['AuditScore'] for rl in ['Low','Medium','High']])
w(f"  F={f:.2f}, p={p:.6f} {'***' if p<0.001 else 'ns'}")

w("\nANOVA: AuditCost by RiskLevel")
for rl in ['Low','Medium','High']:
    w(f"  {rl}: mean=${audit_data[audit_data.RiskLevel==rl]['AuditCost'].mean():.0f}")
f,p = stats.f_oneway(*[audit_data[audit_data.RiskLevel==rl]['AuditCost'] for rl in ['Low','Medium','High']])
w(f"  F={f:.2f}, p={p:.6f} {'***' if p<0.001 else 'ns'}")

w("\n" + "-"*60 + "\n  3. BAYESIAN ANALYSIS\n" + "-"*60)
a_prior,b_prior=1,1

for rl in ['Low','Medium','High']:
    sub = audit_data[audit_data.RiskLevel==rl]; k = int((sub.AuditStatus=='Completed').sum()); n = len(sub)
    a,b = a_prior+k, b_prior+(n-k)
    lo,hi = beta_dist.ppf(0.025,a,b), beta_dist.ppf(0.975,a,b)
    w(f"\nP(Completed | Risk={rl}): Beta({a},{b}) = {a/(a+b)*100:.1f}% [{lo*100:.1f}%-{hi*100:.1f}%]")

w("\nBayesian: AuditType -> High Risk probability")
for at in audit_data.AuditType.unique():
    sub = audit_data[audit_data.AuditType==at]; k = int((sub.RiskLevel=='High').sum()); n = len(sub)
    a,b = a_prior+k, b_prior+(n-k); lo,hi = beta_dist.ppf(0.025,a,b), beta_dist.ppf(0.975,a,b)
    w(f"  {at:15s}: {k:3d}/{n:4d} -> {a/(a+b)*100:.1f}% [{lo*100:.1f}%-{hi*100:.1f}%]")

w("\nBayesian: EncryptionStandard -> High Risk")
for enc in audit_data['EncryptionStandard'].dropna().unique():
    sub = audit_data[audit_data.EncryptionStandard==enc] if 'EncryptionStandard' in audit_data.columns else audit_data
    w(f"  {enc:15s}")

w("\n" + "-"*60 + "\n  4. VISUALIZATIONS\n" + "-"*60)
fig, axes = plt.subplots(2,3,figsize=(18,12))
fig.suptitle('Project 3: Audit & Security Analysis', fontsize=16, fontweight='bold')

audit_data['RiskLevel'].value_counts().plot(kind='bar',ax=axes[0,0],color=['#2ecc71','#f39c12','#e74c3c'],edgecolor='k')
axes[0,0].set_title('Risk Level Distribution',fontweight='bold')

audit_data['AuditType'].value_counts().plot(kind='bar',ax=axes[0,1],color=['#3498db','#9b59b6','#1abc9c','#e67e22','#34495e'],edgecolor='k')
axes[0,1].set_title('Audit Type Distribution',fontweight='bold')

pd.crosstab(audit_data['AuditType'],audit_data['RiskLevel'],normalize='index').plot(kind='bar',stacked=True,ax=axes[0,2],color=['#2ecc71','#f39c12','#e74c3c'],edgecolor='k')
axes[0,2].set_title('Risk Level by Audit Type',fontweight='bold')

pd.crosstab(audit_data['AuditStatus'],audit_data['RiskLevel'],normalize='index').plot(kind='bar',stacked=True,ax=axes[1,0],color=['#2ecc71','#f39c12','#e74c3c'],edgecolor='k')
axes[1,0].set_title('Risk Level by Status',fontweight='bold')

audit_data.boxplot(column='AuditScore',by='RiskLevel',ax=axes[1,1]); axes[1,1].set_title('AuditScore by RiskLevel',fontweight='bold')
audit_data.boxplot(column='AuditCost',by='RiskLevel',ax=axes[1,2]); axes[1,2].set_title('AuditCost by RiskLevel',fontweight='bold')

plt.tight_layout()
fig.savefig(f"{OUT}\\Audit_Project.png",dpi=150,bbox_inches='tight')
w("  [OK] Audit_Project.png")

log.close()
print(f"\nPROJECT 3 COMPLETE -> {OUT}\\report.txt")
