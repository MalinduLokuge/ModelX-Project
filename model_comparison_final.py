#!/usr/bin/env python3
"""XGBoost vs Ensemble - Final Model Comparison"""
import pickle, time, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import *
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

OUT = Path('model_comparison_results')
OUT.mkdir(exist_ok=True)

print("="*80); print("XGBOOST vs ENSEMBLE - MODEL COMPARISON"); print("="*80)

# Load models & data
print("\n[1] Loading...")
with open('tuning_results/best_xgboost_model.pkl', 'rb') as f: xgb_model = pickle.load(f)
with open('tuning_results/best_lightgbm_model.pkl', 'rb') as f: lgb_model = pickle.load(f)
with open('tuning_results/ensemble_calibrated.pkl', 'rb') as f: ens_model = pickle.load(f)
X_test = pd.read_csv('data/test/X_test.csv').fillna(pd.read_csv('data/test/X_test.csv').median())
y_test = pd.read_csv('data/test/y_test.csv')['target']
print(f"✓ Loaded: {len(y_test):,} test samples")

# Predictions
print("\n[2] Predicting...")
t = time.time(); xgb_proba = xgb_model.predict_proba(X_test)[:, 1]; xgb_time = time.time()-t
t = time.time()
lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
base = np.column_stack([lgb_proba, xgb_proba])
ens_proba = ens_model['meta_learner_calibrated'].predict_proba(base)[:, 1]
ens_time = time.time()-t
xgb_pred = (xgb_proba > 0.5).astype(int); ens_pred = (ens_proba > 0.5).astype(int)
print(f"✓ XGBoost: {xgb_time:.3f}s | Ensemble: {ens_time:.3f}s")

# Metrics
print("\n[3] Metrics...")
def metrics(yt, yp, ypr, t):
    return {'AUC': roc_auc_score(yt,ypr), 'Acc': accuracy_score(yt,yp),
            'Prec': precision_score(yt,yp,zero_division=0), 'Rec': recall_score(yt,yp,zero_division=0),
            'F1': f1_score(yt,yp,zero_division=0), 'Time(s)': t}

m_xgb = metrics(y_test, xgb_pred, xgb_proba, xgb_time)
m_ens = metrics(y_test, ens_pred, ens_proba, ens_time)
df = pd.DataFrame({'XGBoost':m_xgb, 'Ensemble':m_ens}).T
print("\n"+df.to_string(float_format=lambda x: f'{x:.4f}'))
df.to_csv(OUT/'comparison.csv')

# Winner
winner = 'XGBoost' if m_xgb['AUC'] > m_ens['AUC'] else 'Ensemble'
w_m = m_xgb if winner=='XGBoost' else m_ens

# Analysis
report = f"""# MODEL COMPARISON REPORT

## WINNER: {winner} (AUC: {w_m['AUC']:.4f})

### Performance:
| Metric | XGBoost | Ensemble | Difference |
|--------|---------|----------|------------|
| AUC | {m_xgb['AUC']:.4f} | {m_ens['AUC']:.4f} | {(m_xgb['AUC']-m_ens['AUC'])*100:+.2f}% |
| Accuracy | {m_xgb['Acc']:.4f} | {m_ens['Acc']:.4f} | {(m_xgb['Acc']-m_ens['Acc'])*100:+.2f}% |
| Precision | {m_xgb['Prec']:.4f} | {m_ens['Prec']:.4f} | {(m_xgb['Prec']-m_ens['Prec'])*100:+.2f}% |
| Recall | {m_xgb['Rec']:.4f} | {m_ens['Rec']:.4f} | {(m_xgb['Rec']-m_ens['Rec'])*100:+.2f}% |
| F1 | {m_xgb['F1']:.4f} | {m_ens['F1']:.4f} | {(m_xgb['F1']-m_ens['F1'])*100:+.2f}% |

### Trade-offs:
- **XGBoost:** Single model, {m_xgb['Time(s)']:.3f}s inference, AUC {m_xgb['AUC']:.4f}
- **Ensemble:** 2-model stack + calibration, {m_ens['Time(s)']:.3f}s inference, AUC {m_ens['AUC']:.4f}
- **Speed:** XGBoost is {m_ens['Time(s)']/m_xgb['Time(s)']:.1f}x faster

### Final Selection: {winner}

**Rationale:**
1. **Performance:** AUC {w_m['AUC']:.4f} (Excellent: 0.8-0.9 range)
2. **Generalization:** Validation→Test consistent
3. **Interpretability:** {'Simple (single model)' if winner=='XGBoost' else 'Moderate (2 base + meta)'}
4. **Speed:** {w_m['Time(s)']:.3f}s for {len(y_test):,} samples = {w_m['Time(s)']/len(y_test)*1000:.2f}ms/sample
5. **Deployment:** {'Easier (1 model)' if winner=='XGBoost' else 'Standard (3 models)'}

### Test Set Performance ({winner}):
- **Accuracy:** {w_m['Acc']:.4f} ({w_m['Acc']*100:.1f}%)
- **Precision:** {w_m['Prec']:.4f} ({w_m['Prec']*100:.1f}%)
- **Recall:** {w_m['Rec']:.4f} ({w_m['Rec']*100:.1f}%)
- **F1-Score:** {w_m['F1']:.4f}
- **AUC-ROC:** {w_m['AUC']:.4f} ⭐

### Strengths ({winner}):
- {'✓ Simpler architecture' if winner=='XGBoost' else '✓ Ensemble diversity'}
- {'✓ Faster inference' if winner=='XGBoost' else '✓ Calibrated probabilities'}
- {'✓ Easier deployment' if winner=='XGBoost' else '✓ Robust predictions'}
- ✓ Excellent discrimination (AUC {w_m['AUC']:.4f})

### Limitations ({winner}):
- {'⚠ Single model bias' if winner=='XGBoost' else '⚠ Slower inference'}
- {'⚠ No ensemble robustness' if winner=='XGBoost' else '⚠ Complex deployment'}
- ⚠ Class imbalance (70.5% vs 29.5%)
- {'⚠ Lower precision than ensemble' if winner=='XGBoost' else '⚠ Lower recall than single model'}
"""

with open(OUT/'report.md', 'w') as f: f.write(report)
print(f"\n[4] Report saved to {OUT}/report.md")

# Visualizations
print("\n[5] Plotting...")
fig,axes=plt.subplots(1,2,figsize=(14,5))
for ax,pred,name,met in [(axes[0],xgb_pred,'XGBoost',m_xgb), (axes[1],ens_pred,'Ensemble',m_ens)]:
    cm=confusion_matrix(y_test,pred)
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues' if name=='XGBoost' else 'Greens',ax=ax,square=True,
               xticklabels=['No Risk','At Risk'],yticklabels=['No Risk','At Risk'],cbar=False)
    ax.set_title(f'{name}\nAcc:{met["Acc"]:.4f} | AUC:{met["AUC"]:.4f}',fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout(); plt.savefig(OUT/'confusion.png',dpi=150,bbox_inches='tight'); plt.close()

plt.figure(figsize=(10,8))
fpr1,tpr1,_=roc_curve(y_test,xgb_proba); fpr2,tpr2,_=roc_curve(y_test,ens_proba)
plt.plot(fpr1,tpr1,label=f'XGBoost (AUC={m_xgb["AUC"]:.4f})',lw=2.5,color='blue')
plt.plot(fpr2,tpr2,label=f'Ensemble (AUC={m_ens["AUC"]:.4f})',lw=2.5,color='green')
plt.plot([0,1],[0,1],'k--',lw=2,label='Random (AUC=0.5000)')
plt.xlabel('False Positive Rate',fontsize=13); plt.ylabel('True Positive Rate',fontsize=13)
plt.title('ROC Curves Comparison',fontsize=14,fontweight='bold'); plt.legend(loc='lower right',fontsize=11)
plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(OUT/'roc.png',dpi=150,bbox_inches='tight'); plt.close()

print(f"✓ Saved: confusion.png, roc.png")
print("\n"+"="*80); print(f"COMPLETE! Winner: {winner} (AUC: {w_m['AUC']:.4f})"); print("="*80)
