
# =============================================================================
# London 分难度合成攻击实验
# Easy / Medium / Hard 三个难度级别，验证各模型在不同攻击隐蔽性下的性能退化
# =============================================================================

import os, time, warnings, gc, json
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = str(min(os.cpu_count() or 4, 8))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from scipy.stats import rankdata, skew, kurtosis
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb

torch.set_num_threads(min(os.cpu_count() or 4, 8))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LONDON_BASE = r'C:\Users\wb.zhoushujie\Desktop\london_smart_meters'
RESULT_DIR = r'C:\Users\wb.zhoushujie\PyCharmMiscProject\results'
os.makedirs(RESULT_DIR, exist_ok=True)

print("=" * 70)
print("London 分难度合成攻击实验 (Easy / Medium / Hard)")
print(f"  设备={device}")
print("=" * 70)
t_total = time.time()

# ─── 三个难度级别的攻击参数定义 ───
DIFFICULTY_CONFIGS = {
    'Easy': {
        'scale_range': (0.1, 0.5),
        'low_val_pct': 10,
        'zero_period': (2, 4),
        'shift_range': (0.4, 0.7),
        'zero_day_ratio': (0.3, 0.5),
        'decay_end': (0.1, 0.3),
    },
    'Medium': {
        'scale_range': (0.5, 0.8),
        'low_val_pct': 25,
        'zero_period': (5, 10),
        'shift_range': (0.15, 0.35),
        'zero_day_ratio': (0.15, 0.25),
        'decay_end': (0.4, 0.6),
    },
    'Hard': {
        'scale_range': (0.8, 0.95),
        'low_val_pct': 40,
        'zero_period': (10, 20),
        'shift_range': (0.05, 0.15),
        'zero_day_ratio': (0.05, 0.12),
        'decay_end': (0.7, 0.9),
    },
}

# ─── Step 1: 加载数据（只做一次） ───
print("\n--- Step 1: 加载 London 数据 ---")
daily = pd.read_csv(os.path.join(LONDON_BASE, 'daily_dataset.csv'),
                     usecols=['LCLid', 'day', 'energy_sum'])
daily['day'] = pd.to_datetime(daily['day'])
all_dates = pd.date_range(daily['day'].min(), daily['day'].max(), freq='D')
T_TOTAL_DAYS = len(all_dates)
MIN_COVERAGE = 0.80
user_day_counts = daily.groupby('LCLid')['day'].nunique()
valid_users = user_day_counts[user_day_counts >= T_TOTAL_DAYS * MIN_COVERAGE].index
daily = daily[daily['LCLid'].isin(valid_users)].copy()
pivot = daily.pivot_table(index='LCLid', columns='day', values='energy_sum', aggfunc='first')
pivot = pivot.reindex(columns=all_dates)
raw_vals_clean = pivot.values.astype(np.float32)
del daily, pivot; gc.collect()

N_USERS, T_DAYS = raw_vals_clean.shape
dates = all_dates
day_of_week = dates.dayofweek.values
month_of_year = dates.month.values
weekday_mask = (day_of_week < 5)
weekend_mask = (day_of_week >= 5)
DAYSPM = 30
NM = T_DAYS // DAYSPM
half_nm = NM // 2
BASELINE_M = 6
print(f"  用户数={N_USERS}, 天数={T_DAYS}, 月度={NM}")


# =============================================================================
# 核心函数定义
# =============================================================================

def inject_attacks(raw_clean, config, theft_ratio=0.10, seed=42):
    """在干净数据副本上注入指定难度的合成攻击，返回 (被攻击数据, 标签)"""
    rng = np.random.RandomState(seed)
    raw = raw_clean.copy()
    N, T = raw.shape
    n_theft = int(N * theft_ratio)
    theft_idx = rng.choice(N, size=n_theft, replace=False)
    labels = np.zeros(N, dtype=np.int64)
    labels[theft_idx] = 1
    attack_start = T // 2
    n_per = n_theft // 6

    for i, uid in enumerate(theft_idx):
        atype = i // n_per if i < n_per * 6 else rng.randint(0, 6)
        seg = raw[uid, attack_start:]
        if atype == 0:  # 随机缩放
            s = rng.uniform(*config['scale_range'])
            raw[uid, attack_start:] = seg * s
        elif atype == 1:  # 固定低值
            pct = config['low_val_pct']
            lv = np.nanpercentile(seg[seg > 0], pct) if (seg > 0).any() else 0.1
            raw[uid, attack_start:] = lv
        elif atype == 2:  # 周期性置零
            p = rng.randint(*config['zero_period'])
            for t in range(0, len(seg), p):
                raw[uid, attack_start + t] = 0.0
        elif atype == 3:  # 均值偏移
            shift = rng.uniform(*config['shift_range']) * np.nanmean(seg)
            raw[uid, attack_start:] = np.maximum(seg - shift, 0)
        elif atype == 4:  # 随机天置零
            zr = rng.uniform(*config['zero_day_ratio'])
            zd = rng.choice(len(seg), size=int(len(seg) * zr), replace=False)
            raw[uid, attack_start + zd] = 0.0
        else:  # 渐进式衰减
            de = rng.uniform(*config['decay_end'])
            decay = np.linspace(1.0, de, len(seg)).astype(np.float32)
            raw[uid, attack_start:] = seg * decay
    return raw, labels


def build_features(raw_vals, labels_arr):
    """完整特征工程 pipeline，返回 (X_baseline, X_g2, X_mo_seq, FEAT_DIM)"""
    N = raw_vals.shape[0]; T = raw_vals.shape[1]
    # 缺失模式
    nan_mask = np.isnan(raw_vals)
    miss_ratio = nan_mask.mean(axis=1).astype(np.float32)
    ht = T // 2
    mfh = nan_mask[:, :ht].mean(axis=1).astype(np.float32)
    msh = nan_mask[:, ht:].mean(axis=1).astype(np.float32)
    mhd = msh - mfh
    mcn = np.zeros(N, dtype=np.float32); cr = np.zeros(N, dtype=np.float32)
    for t in range(T):
        cr = (cr + 1.0) * nan_mask[:, t]; mcn = np.maximum(mcn, cr)
    ni = nan_mask.astype(np.int32); nd = np.diff(ni, axis=1, prepend=0)
    msc = (nd == 1).sum(axis=1).astype(np.float32)
    zm = (raw_vals == 0) & (~nan_mask)
    zr = zm.sum(axis=1).astype(np.float32) / (~nan_mask).sum(axis=1).clip(1).astype(np.float32)
    mcz = np.zeros(N, dtype=np.float32); crz = np.zeros(N, dtype=np.float32)
    for t in range(T):
        crz = (crz + 1.0) * zm[:, t]; mcz = np.maximum(mcz, crz)
    nsr = (nan_mask | zm).mean(axis=1).astype(np.float32)
    miss_f = np.column_stack([miss_ratio, mcn/T, mfh, msh, mhd, msc, zr, mcz/T, nsr]).astype(np.float32)

    # 插值
    col_m = np.nan_to_num(np.nanmean(raw_vals, axis=0), nan=0.0)
    for i in range(0, N, 5000):
        c = pd.DataFrame(raw_vals[i:i+5000])
        c = c.interpolate(method='linear', axis=1, limit_direction='both').fillna(pd.Series(col_m))
        raw_vals[i:i+5000] = c.values.astype(np.float32)
    X = raw_vals

    # 月度聚合
    mo_mean = np.zeros((N, NM), dtype=np.float32)
    mo_std  = np.zeros((N, NM), dtype=np.float32)
    mo_max  = np.zeros((N, NM), dtype=np.float32)
    mo_zero = np.zeros((N, NM), dtype=np.float32)
    mo_nan  = np.zeros((N, NM), dtype=np.float32)
    for m in range(NM):
        s, e = m*DAYSPM, min((m+1)*DAYSPM, T)
        mo_mean[:,m] = X[:,s:e].mean(1); mo_std[:,m] = X[:,s:e].std(1)
        mo_max[:,m] = X[:,s:e].max(1); mo_zero[:,m] = (X[:,s:e]==0).mean(1)
        mo_nan[:,m] = nan_mask[:,s:e].mean(1)

    bl = mo_mean[:,:BASELINE_M].mean(1,keepdims=True)+1e-3
    mo_vs = (mo_mean - bl)/(np.abs(bl)+1e-3)
    mo_cum = np.cumsum(mo_mean - bl, axis=1)
    mo_pct = np.zeros((N,NM),dtype=np.float32)
    for m in range(NM): mo_pct[:,m] = (rankdata(mo_mean[:,m])/N).astype(np.float32)
    mo_rm = mo_pct.mean(1,keepdims=True); mo_rd = mo_pct - mo_rm
    rk_drop = mo_pct[:,:half_nm].mean(1) - mo_pct[:,half_nm:].mean(1)
    u_med = np.median(mo_mean,1,keepdims=True)+1e-3
    mo_sr = mo_mean/u_med
    mo_lr = np.log1p(np.maximum(mo_mean,0)) - np.log1p(np.maximum(bl,0))
    mo_r3m = np.zeros((N,NM),dtype=np.float32); mo_r3s = np.zeros((N,NM),dtype=np.float32)
    for m in range(NM):
        ws = max(0,m-2); mo_r3m[:,m]=mo_mean[:,ws:m+1].mean(1); mo_r3s[:,m]=mo_mean[:,ws:m+1].std(1)+1e-6
    mo_lz = np.clip((mo_mean-mo_r3m)/mo_r3s,-5,5)
    mo_gm = np.median(mo_mean,0,keepdims=True)+1e-3
    mo_gd = np.log1p(np.maximum(mo_mean,0)) - np.log1p(np.maximum(mo_gm,0))
    mo_d1 = np.diff(mo_mean,axis=1,prepend=mo_mean[:,:1])
    mo_d2 = np.diff(mo_d1,axis=1,prepend=mo_d1[:,:1])
    mo_cv = mo_std/(mo_mean+1e-6)

    # 日历
    wd_m = np.nan_to_num(np.nanmean(np.where(weekday_mask[None,:],X,np.nan),1),0).astype(np.float32)
    we_m = np.nan_to_num(np.nanmean(np.where(weekend_mask[None,:],X,np.nan),1),0).astype(np.float32)
    wd_s = np.nan_to_num(np.nanstd(np.where(weekday_mask[None,:],X,np.nan),1),0).astype(np.float32)
    we_s = np.nan_to_num(np.nanstd(np.where(weekend_mask[None,:],X,np.nan),1),0).astype(np.float32)
    mac = np.zeros((N,12),dtype=np.float32)
    for mc in range(12):
        mm = (month_of_year==mc+1)
        if mm.sum()>0: mac[:,mc] = np.nan_to_num(np.nanmean(np.where(mm[None,:],X,np.nan),1),0).astype(np.float32)
    sa = mac[:,[5,6,7]].mean(1); wa = mac[:,[11,0,1]].mean(1)
    swr = sa/(wa+1e-6); ya = mac.mean(1,keepdims=True)+1e-6; ss = (mac/ya).std(1)
    mo_wdm = np.zeros((N,NM),dtype=np.float32); mo_wem = np.zeros((N,NM),dtype=np.float32)
    for m in range(NM):
        s,e = m*DAYSPM, min((m+1)*DAYSPM,T)
        wm_ = weekday_mask[s:e]; wem_ = weekend_mask[s:e]
        if wm_.sum()>0: mo_wdm[:,m] = X[:,s:e][:,wm_].mean(1)
        if wem_.sum()>0: mo_wem[:,m] = X[:,s:e][:,wem_].mean(1)
    mo_wwr = mo_wdm/(mo_wem+1e-6)
    cal_f = np.column_stack([wd_m/(we_m+1e-6),wd_m,we_m,wd_s,we_s,swr,ss]).astype(np.float32)

    # 深层统计
    um=X.mean(1);us=X.std(1);umd=np.median(X,1);umx=X.max(1)
    uq10=np.percentile(X,10,1);uq25=np.percentile(X,25,1)
    uq75=np.percentile(X,75,1);uq90=np.percentile(X,90,1)
    uiqr=uq75-uq25;ucv=us/(um+1e-6)
    usk=np.nan_to_num(skew(X,1,nan_policy='omit').astype(np.float32),0)
    ukt=np.nan_to_num(kurtosis(X,1,nan_policy='omit').astype(np.float32),0)
    eh=(X>(uq75+3*uiqr)[:,None]).mean(1).astype(np.float32)
    el=(X<np.maximum(uq25-3*uiqr,0)[:,None]).mean(1).astype(np.float32)
    tv=np.arange(T,dtype=np.float64);tmv=tv.mean();tvv=((tv-tmv)**2).sum()
    Xf=X.astype(np.float64)
    sl=((Xf-Xf.mean(1,keepdims=True))*(tv[None,:]-tmv)).sum(1)/(tvv+1e-8)
    del Xf
    thr=(mo_mean[:,-6:].mean(1)/(mo_mean[:,:6].mean(1)+1e-6)).astype(np.float32)
    mrt=mo_mean[:,1:]/(mo_mean[:,:-1]+1e-6)
    dmt=(mrt<0.7).astype(np.int32)
    mcd=np.zeros(N,dtype=np.float32);cdr=np.zeros(N,dtype=np.float32)
    for t in range(dmt.shape[1]):
        cdr=(cdr+1.0)*dmt[:,t];mcd=np.maximum(mcd,cdr)
    mq25=np.zeros((N,NM),dtype=np.float32);mq75=np.zeros((N,NM),dtype=np.float32)
    for m in range(NM):
        s,e=m*DAYSPM,min((m+1)*DAYSPM,T)
        mq25[:,m]=np.percentile(X[:,s:e],25,1);mq75[:,m]=np.percentile(X[:,s:e],75,1)
    q25t=mq25[:,half_nm:].mean(1)-mq25[:,:half_nm].mean(1)
    q75t=mq75[:,half_nm:].mean(1)-mq75[:,:half_nm].mean(1)
    stt=mo_std[:,half_nm:].mean(1)/(mo_std[:,:half_nm].mean(1)+1e-6)
    cvt=mo_cv[:,half_nm:].mean(1)-mo_cv[:,:half_nm].mean(1)
    ds_f = np.column_stack([um,us,umd,umx,uq10,uq25,uq75,uq90,uiqr,ucv,usk,ukt,eh,el,
                            sl.astype(np.float32),thr,mcd,q25t,q75t,stt,cvt]).astype(np.float32)

    # TCN-CPD
    ti = X.copy().astype(np.float32)
    _um=np.median(ti,1,keepdims=True);_uiq=np.clip(np.percentile(ti,75,1,keepdims=True)-np.percentile(ti,25,1,keepdims=True),1e-6,None)
    ti = np.clip((ti-_um)/_uiq,-10,10); _Tt=min(ti.shape[1],1020); ti=ti[:,:_Tt]
    ms_cpd=[]
    for sc in [7,30,90]:
        k=np.ones(sc)/sc; sm=np.apply_along_axis(lambda r:np.convolve(r,k,'same'),1,ti)
        ar=np.abs(ti-sm); ns=max(_Tt//sc,4); sl2=_Tt//ns
        se=np.zeros((N,ns),dtype=np.float32)
        for si in range(ns):
            s2,e2=si*sl2,min((si+1)*sl2,_Tt); se[:,si]=ar[:,s2:e2].mean(1)
        sd=np.abs(np.diff(se,axis=1))
        mjp=np.argmax(sd,1).astype(np.float32)/max(ns-2,1)
        mjm=np.max(sd,1)/(se.mean(1)+1e-8)
        ncp=(sd>2*(sd.mean(1,keepdims=True)+1e-8)).sum(1).astype(np.float32)
        hs=ns//2; hr=(se[:,hs:].mean(1)+1e-8)/(se[:,:hs].mean(1)+1e-8)
        es=se.std(1); xm=np.arange(ns,dtype=np.float32);xmm=xm.mean()
        em=se.mean(1,keepdims=True); _sl2=((xm[None,:]-xmm)*(se-em)).sum(1)/(((xm-xmm)**2).sum()+1e-8)
        ms_cpd.append(np.column_stack([mjp,mjm,ncp,hr,es,_sl2]).astype(np.float32))
    tcn_f=np.concatenate(ms_cpd,axis=1)
    cf=np.column_stack([ms_cpd[0][:,4]/(ms_cpd[2][:,4]+1e-8),np.abs(ms_cpd[0][:,0]-ms_cpd[2][:,0])]).astype(np.float32)
    tcn_f=np.concatenate([tcn_f,cf],axis=1)

    # ISCT
    uag=X.mean(1); stl=pd.qcut(uag,q=8,labels=False,duplicates='drop')
    isct_dm=np.zeros((N,NM),dtype=np.float32)
    for ks in range(int(stl.max())+1):
        ms=(stl==ks)
        if ms.sum()<2: continue
        lm=np.median(mo_mean[ms],0,keepdims=True)
        isct_dm[ms]=((mo_mean[ms]-lm)/(np.abs(lm)+1e-3)).astype(np.float32)
    sa2=isct_dm; Ts=sa2.shape[1]
    icf=np.zeros((N,8),dtype=np.float32)
    sm2=sa2.mean(1,keepdims=True);cs2=np.cumsum(sa2-sm2,1)
    cpi=np.argmax(np.abs(cs2),1); icf[:,0]=cpi.astype(np.float32)/Ts
    cpc=np.clip(cpi,1,Ts-1); at=np.arange(Ts)[None,:]; ce=cpc[:,None]
    pm=(sa2*(at<ce)).sum(1)/((at<ce).sum(1).astype(np.float32)+1e-8)
    po=(sa2*(at>=ce)).sum(1)/((at>=ce).sum(1).astype(np.float32)+1e-8)
    icf[:,1]=(po/(np.abs(pm)+1e-6)).astype(np.float32)
    icf[:,2]=np.abs(sa2).max(1); h2=Ts//2
    icf[:,3]=sa2[:,h2:].mean(1)/(np.abs(sa2[:,:h2].mean(1))+1e-6)
    icf[:,4]=sa2.std(1)
    nm2=(sa2<0).astype(np.int32);mxn=np.zeros(N,dtype=np.float32);cr3=np.zeros(N,dtype=np.float32)
    for ts in range(Ts): cr3=(cr3+1.0)*nm2[:,ts]; mxn=np.maximum(mxn,cr3)
    icf[:,5]=mxn/Ts
    tvs=np.arange(Ts,dtype=np.float64);tmv2=tvs.mean();tvv2=((tvs-tmv2)**2).sum()
    sf=sa2.astype(np.float64)
    icf[:,6]=((sf-sf.mean(1,keepdims=True))*(tvs[None,:]-tmv2)).sum(1)/(tvv2+1e-8)
    icf[:,7]=sa2[:,-6:].mean(1)
    ik=[c for c in range(8) if abs(roc_auc_score(labels_arr,icf[:,c])-0.5)>0.03]
    icf=icf[:,ik] if ik else icf

    # RMT
    mnf=mo_mean.astype(np.float64)
    mg=np.median(mnf,1,keepdims=True);miq=np.clip(np.percentile(mnf,75,1,keepdims=True)-np.percentile(mnf,25,1,keepdims=True),1e-6,None)
    mng=np.clip((mnf-mg)/miq,-5,5); Xc=mng-mng.mean(0,keepdims=True)
    cov=(Xc.T@Xc)/(N-1); ev,evc=np.linalg.eigh(cov); ev=np.maximum(ev,0)
    s2=np.median(ev[ev>0]) if (ev>0).sum()>0 else 1.0
    r=NM/max(N,NM+1); mpu=s2*(1+np.sqrt(r))**2
    sm3=ev>mpu;nm3=~sm3
    ps=Xc@evc[:,sm3] if sm3.sum()>0 else np.zeros((N,1))
    pn=Xc@evc[:,nm3] if nm3.sum()>0 else np.ones((N,1))
    Bsnr=((ps**2).sum(1)/((pn**2).sum(1)+1e-8)).astype(np.float32)
    stlr=pd.qcut(uag,q=8,labels=False,duplicates='drop')
    Alate=np.zeros(N,dtype=np.float32)
    for ks in range(int(stlr.max())+1):
        mk=(stlr==ks); nl=mk.sum()
        if nl<20: continue
        ld=mng[mk]; lc=ld-ld.mean(0,keepdims=True)
        cl=(lc.T@lc)/(nl-1); el,ecl=np.linalg.eigh(cl); el=np.maximum(el,0)
        s2l=np.median(el[el>0]) if (el>0).sum()>0 else 1.0
        rl=NM/max(nl,NM+1); mpl=s2l*(1+np.sqrt(rl))**2
        sml=el>mpl
        if sml.sum()==0: continue
        Vs=ecl[:,sml]; ls=NM*2//3
        fp=(lc@Vs)**2; ltc=np.zeros_like(lc); ltc[:,ls:]=lc[:,ls:]
        lp=(ltc@Vs)**2
        Alate[mk]=(lp.sum(1)/(fp.sum(1)+1e-8)).astype(np.float32)
    rmt_f=np.column_stack([Bsnr,Alate]).astype(np.float32)

    # 组装
    fb=[mo_mean,mo_std,mo_max,mo_zero,mo_nan,mo_vs,mo_cum,mo_pct,mo_rd,
        mo_sr,mo_lr,mo_r3m,mo_r3s,mo_lz,mo_gd,mo_d1,mo_d2,mo_cv,mo_wwr,
        rk_drop[:,None],miss_f,cal_f,ds_f,tcn_f,icf]
    X_bl=np.concatenate(fb,axis=1).astype(np.float32)
    X_g2=np.concatenate([X_bl,rmt_f],axis=1).astype(np.float32)

    # Transformer 序列
    def _sc(a,c=5.0):
        f=a.reshape(-1,1); f=RobustScaler().fit_transform(f).reshape(a.shape)
        return np.clip(f,-c,c).astype(np.float32)
    def _scu(a,c=5.0):
        md=np.median(a,1,keepdims=True);q1=np.percentile(a,25,1,keepdims=True)
        q3=np.percentile(a,75,1,keepdims=True);iq=q3-q1+1e-6
        return np.clip((a-md)/iq,-c,c).astype(np.float32)
    rt=np.tile(rk_drop[:,np.newaxis],(1,NM)); rdc=mo_mean-mo_gm
    mch=np.stack([_scu(mo_mean),_scu(mo_std),_scu(mo_max),mo_zero,mo_pct,
                  _sc(rdc),_sc(rt),_scu(mo_vs),_scu(mo_cum),_sc(mo_lr),
                  _scu(mo_d1),_sc(mo_d2),mo_sr.astype(np.float32),
                  np.clip(mo_lz,-5,5).astype(np.float32),_sc(mo_gd),
                  _scu(mo_mean.copy()),_scu(isct_dm)],axis=2)
    sf2=mch.mean(1).astype(np.float32)
    st2=np.tile(sf2[:,np.newaxis,:],(1,NM,1))
    Xms=np.concatenate([mch,st2],axis=2).astype(np.float32)
    FD=Xms.shape[2]
    return X_bl, X_g2, Xms, FD


# ── DualPathTransformer（同 london_phase3_transformer.py） ──

class LocalWindowAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, win_size=4, dim_ff=256, dropout=0.1):
        super().__init__()
        self.win_size = win_size
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model,dim_ff),nn.GELU(),nn.Dropout(dropout),
                                 nn.Linear(dim_ff,d_model),nn.Dropout(dropout))
    def _build_mask(self, L, dev):
        if hasattr(self,'_mc') and self._mc.shape[0]==L: return self._mc.to(dev)
        m=torch.full((L,L),float('-inf'))
        for i in range(L):
            s,e=max(0,i-self.win_size),min(L,i+self.win_size+1); m[i,s:e]=0.0
        self._mc=m; return m.to(dev)
    def forward(self, x):
        h=self.norm1(x); m=self._build_mask(x.size(1),x.device)
        x=x+self.attn(h,h,h,attn_mask=m)[0]; x=x+self.ffn(self.norm2(x)); return x

class DualPathTransformer(nn.Module):
    def __init__(self, feat_dim=34, d_model=128, nhead=4, num_layers=3,
                 dim_ff=256, dropout=0.15, win_size=4, max_len=40):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(feat_dim,d_model),nn.LayerNorm(d_model))
        pe=torch.zeros(max_len,d_model)
        pos=torch.arange(0,max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d_model,2).float()*(-np.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div[:d_model//2])
        self.register_buffer('pe',pe.unsqueeze(0))
        self.pe_drop=nn.Dropout(dropout)
        self.local_layers=nn.ModuleList([LocalWindowAttentionBlock(d_model,nhead,win_size,dim_ff,dropout) for _ in range(num_layers)])
        self.cls_token=nn.Parameter(torch.zeros(1,1,d_model)); nn.init.trunc_normal_(self.cls_token,std=0.02)
        el=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dim_ff,
                                      dropout=dropout,batch_first=True,norm_first=True)
        self.global_transformer=nn.TransformerEncoder(el,num_layers=num_layers,enable_nested_tensor=False)
        self.classifier=nn.Sequential(nn.LayerNorm(d_model*2),
            nn.Linear(d_model*2,d_model),nn.GELU(),nn.Dropout(0.3),
            nn.Linear(d_model,d_model//2),nn.GELU(),nn.Dropout(0.3),nn.Linear(d_model//2,1))
    def _encode(self, x):
        B,L,_=x.shape; h=self.pe_drop(self.input_proj(x)+self.pe[:,:L,:])
        hl=h
        for bl in self.local_layers: hl=bl(hl)
        fl=hl.mean(dim=1)
        hg=torch.cat([self.cls_token.expand(B,-1,-1),h],dim=1)
        fg=self.global_transformer(hg)[:,0,:]; return fl,fg
    def forward(self, x):
        fl,fg=self._encode(x); fc=torch.cat([fl,fg],1); return self.classifier(fc),fc
    def extract_features(self, x):
        fl,fg=self._encode(x); return torch.cat([fl,fg],1)

class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=0.92, gamma=2.0):
        super().__init__(); self.alpha,self.gamma=alpha,gamma
    def forward(self, logits, targets):
        p=torch.sigmoid(logits)
        bce=nn.functional.binary_cross_entropy_with_logits(logits,targets,reduction='none')
        pt=p*targets+(1-p)*(1-targets); f=(1-pt)**self.gamma*bce
        at=self.alpha*targets+(1-self.alpha)*(1-targets); return (at*f).mean()


def train_transformer(X_mo_seq, labels_arr, feat_dim, epochs=40, patience=8):
    """训练 Transformer 并返回 (best_auc, trans_pca16 特征)"""
    np.random.seed(42); torch.manual_seed(42)
    idx_tr,idx_te=train_test_split(np.arange(len(labels_arr)),test_size=0.2,random_state=42,stratify=labels_arr)
    Xtr=torch.FloatTensor(X_mo_seq[idx_tr]); ytr=torch.FloatTensor(labels_arr[idx_tr].astype(np.float32))
    Xte=torch.FloatTensor(X_mo_seq[idx_te]); yte=torch.FloatTensor(labels_arr[idx_te].astype(np.float32))
    cc=np.bincount(labels_arr[idx_tr]); sw=(1.0/cc)[labels_arr[idx_tr]]
    sampler=WeightedRandomSampler(torch.FloatTensor(sw),len(idx_tr),replacement=True)
    tl=DataLoader(TensorDataset(Xtr,ytr),batch_size=1024,sampler=sampler,num_workers=0)
    vl=DataLoader(TensorDataset(Xte,yte),batch_size=2048,shuffle=False,num_workers=0)
    pc=int((labels_arr==1).sum()); nc=int((labels_arr==0).sum())
    aa=max((1.0-pc/(pc+nc))*0.95,0.80)
    mdl=DualPathTransformer(feat_dim=feat_dim,d_model=128,nhead=4,num_layers=3,dim_ff=256,
                            dropout=0.15,win_size=4,max_len=40).to(device)
    crit=AdaptiveFocalLoss(alpha=aa,gamma=2.0)
    opt=optim.AdamW(mdl.parameters(),lr=1e-3,weight_decay=1e-4)
    sched=optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=20,T_mult=2,eta_min=5e-7)
    best_auc,best_st,best_ep,ni=0.0,None,0,0
    for ep in range(1,epochs+1):
        mdl.train(); tl_sum,tl_n=0.0,0
        for xb,yb in tl:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad(set_to_none=True)
            lg,_=mdl(xb); loss=crit(lg.squeeze(1),yb); loss.backward()
            nn.utils.clip_grad_norm_(mdl.parameters(),1.0); opt.step()
            tl_sum+=loss.item()*len(yb); tl_n+=len(yb)
        mdl.eval(); ap,al=[],[]
        with torch.no_grad():
            for xb,yb in vl:
                lg,_=mdl(xb.to(device)); ap.extend(torch.sigmoid(lg.squeeze(1)).cpu().numpy()); al.extend(yb.numpy())
        va=roc_auc_score(np.array(al),np.array(ap))
        sched.step(ep-1)
        if va>best_auc:
            best_auc=va; best_st={k:v.cpu().clone() for k,v in mdl.state_dict().items()}; best_ep=ep; ni=0
        else:
            ni+=1
            if ni>=patience: break
    mdl.load_state_dict(best_st)
    # 提取特征
    mdl.eval(); feats=[]
    with torch.no_grad():
        for i in range(0,len(X_mo_seq),2048):
            xb=torch.FloatTensor(X_mo_seq[i:i+2048]).to(device)
            feats.append(mdl.extract_features(xb).cpu().numpy())
    f256=np.concatenate(feats,0)
    pca=PCA(n_components=16,random_state=42)
    f16=pca.fit_transform(f256).astype(np.float32)
    return best_auc, f16


def _best_f1(y_true, scores):
    prec, rec, _ = precision_recall_curve(y_true, scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    return f1s.max()


def evaluate_all_models(X_g3, labels_arr, trans_auc):
    """对 G3 特征跑所有模型的 5折 OOF，返回字典 {model_name: {auc, f1}}"""
    N = len(labels_arr)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    RS = RobustScaler()
    X_sc = RS.fit_transform(X_g3)
    pca32 = PCA(n_components=32, random_state=42)
    X_pca = pca32.fit_transform(X_sc).astype(np.float32)

    oof = {k: np.zeros(N, dtype=np.float64) for k in
           ['cat','xgb','lgb','rf','lr','mlp','if']}

    for fi,(tri,vai) in enumerate(skf.split(X_g3, labels_arr)):
        ytr,yva = labels_arr[tri], labels_arr[vai]
        pw = (ytr==0).sum()/max((ytr==1).sum(),1)

        cat=CatBoostClassifier(iterations=1500,depth=6,learning_rate=0.03,l2_leaf_reg=5.0,
                               random_seed=42+fi,auto_class_weights='Balanced',verbose=0,
                               eval_metric='AUC',early_stopping_rounds=100)
        cat.fit(X_g3[tri],ytr,eval_set=(X_g3[vai],yva),verbose=0)
        oof['cat'][vai]=cat.predict_proba(X_g3[vai])[:,1]

        xm=xgb.XGBClassifier(n_estimators=1500,max_depth=6,learning_rate=0.03,
                              reg_alpha=0.5,reg_lambda=2.0,scale_pos_weight=pw,
                              subsample=0.8,colsample_bytree=0.6,tree_method='hist',
                              random_state=42+fi,eval_metric='auc',early_stopping_rounds=100,verbosity=0)
        xm.fit(X_g3[tri],ytr,eval_set=[(X_g3[vai],yva)],verbose=False)
        oof['xgb'][vai]=xm.predict_proba(X_g3[vai])[:,1]

        lm=lgb.LGBMClassifier(n_estimators=1000,max_depth=-1,num_leaves=63,learning_rate=0.03,
                               subsample=0.8,colsample_bytree=0.7,min_child_samples=20,
                               reg_alpha=0.1,reg_lambda=1.0,scale_pos_weight=pw,
                               metric='auc',random_state=42+fi,n_jobs=-1,verbose=-1)
        lm.fit(X_g3[tri],ytr,eval_set=[(X_g3[vai],yva)],callbacks=[lgb.early_stopping(30,verbose=False)])
        oof['lgb'][vai]=lm.predict_proba(X_g3[vai])[:,1]

        rf=RandomForestClassifier(n_estimators=300,max_depth=12,class_weight='balanced',
                                   random_state=42+fi,n_jobs=-1)
        rf.fit(X_g3[tri],ytr); oof['rf'][vai]=rf.predict_proba(X_g3[vai])[:,1]

        lr=LogisticRegression(C=1.0,max_iter=1000,class_weight='balanced',solver='lbfgs',random_state=42)
        lr.fit(X_sc[tri],ytr); oof['lr'][vai]=lr.predict_proba(X_sc[vai])[:,1]

        mlp=MLPClassifier(hidden_layer_sizes=(128,64),activation='relu',max_iter=200,
                           early_stopping=True,validation_fraction=0.1,random_state=42+fi)
        mlp.fit(X_sc[tri],ytr); oof['mlp'][vai]=mlp.predict_proba(X_sc[vai])[:,1]

        iso=IsolationForest(n_estimators=300,contamination=0.1,random_state=42+fi,n_jobs=-1)
        iso.fit(X_sc[tri]); oof['if'][vai]=-iso.decision_function(X_sc[vai])

    # Ensemble — OOF AUC 加权 Rank 融合（弱模型自动降权）
    from scipy.stats import rankdata as _rd
    auc_cat = roc_auc_score(labels_arr, oof['cat'])
    auc_xgb = roc_auc_score(labels_arr, oof['xgb'])
    auc_lgb = roc_auc_score(labels_arr, oof['lgb'])
    w_cat = auc_cat ** 2; w_xgb = auc_xgb ** 2; w_lgb = auc_lgb ** 2
    w_sum = w_cat + w_xgb + w_lgb
    oof_ens = (w_cat * _rd(oof['cat']) + w_xgb * _rd(oof['xgb']) + w_lgb * _rd(oof['lgb'])) / w_sum

    results = {}
    for name, arr in [('Ours Ensemble',oof_ens),('CatBoost',oof['cat']),
                       ('XGBoost',oof['xgb']),('LightGBM',oof['lgb']),
                       ('Random Forest',oof['rf']),('Logistic Reg.',oof['lr']),
                       ('MLP',oof['mlp']),('Isolation Forest',oof['if'])]:
        auc = roc_auc_score(labels_arr, arr)
        if auc < 0.5 and name == 'Isolation Forest':
            arr = -arr; auc = 1 - auc
        f1 = _best_f1(labels_arr, arr)
        results[name] = {'auc': round(auc, 4), 'f1': round(f1, 4)}

    results['Transformer'] = {'auc': round(trans_auc, 4), 'f1': None}
    return results


# =============================================================================
# 主循环：三个难度级别
# =============================================================================

all_results = {}  # {difficulty: {model: {auc, f1}}}

for diff_name, diff_cfg in DIFFICULTY_CONFIGS.items():
    t_diff = time.time()
    print(f"\n{'='*70}")
    print(f"  难度级别: {diff_name}")
    print(f"  参数: scale={diff_cfg['scale_range']}, shift={diff_cfg['shift_range']}, "
          f"zero_ratio={diff_cfg['zero_day_ratio']}, decay_end={diff_cfg['decay_end']}")
    print(f"{'='*70}")

    # 1. 注入攻击
    raw_attacked, labels = inject_attacks(raw_vals_clean, diff_cfg, theft_ratio=0.10, seed=42)
    print(f"  注入完成: 窃电={labels.sum()}, 正常={(labels==0).sum()}")

    # 2. 特征工程
    print(f"  特征工程中...")
    X_bl, X_g2, X_ms, FD = build_features(raw_attacked, labels)
    print(f"  Baseline={X_bl.shape[1]}维, G2={X_g2.shape[1]}维, Trans输入=({X_ms.shape})")

    # 3. Transformer
    print(f"  Transformer 训练中...")
    trans_auc, trans_pca16 = train_transformer(X_ms, labels, FD, epochs=40, patience=8)
    print(f"  Transformer AUC={trans_auc:.4f}")

    # 4. G3 = G2 + Trans_PCA16
    X_g3 = np.concatenate([X_g2, trans_pca16], axis=1).astype(np.float32)
    print(f"  G3={X_g3.shape[1]}维")

    # 5. 所有模型评估
    print(f"  模型评估中...")
    results = evaluate_all_models(X_g3, labels, trans_auc)
    all_results[diff_name] = results

    elapsed = time.time() - t_diff
    print(f"\n  [{diff_name}] 结果 (按AUC降序):")
    print(f"  {'Model':<20} {'AUC':>8} {'F1':>8}")
    print(f"  {'-'*38}")
    sorted_models = sorted(results.keys(), key=lambda m: results[m]['auc'], reverse=True)
    for mname in sorted_models:
        r = results[mname]
        f1_str = f"{r['f1']:.4f}" if r['f1'] is not None else '   -'
        print(f"  {mname:<20} {r['auc']:>8.4f} {f1_str:>8}")
    print(f"  耗时: {elapsed/60:.1f} min")

    del raw_attacked, labels, X_bl, X_g2, X_ms, X_g3, trans_pca16
    gc.collect()

# ─── 保存结果 ───
result_path = os.path.join(RESULT_DIR, 'london_difficulty_results.json')
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存到: {result_path}")

# ─── 汇总表 ───
all_models = list(all_results['Easy'].keys())

# AUC 汇总表 — 按 Hard AUC 从大到小排序
auc_sorted = sorted(all_models, key=lambda m: all_results['Hard'][m]['auc'], reverse=True)
print("\n" + "=" * 100)
print("           London 分难度合成攻击 — AUC 对比汇总 (按Hard AUC降序)")
print("=" * 100)
print(f"{'Model':<20} {'Easy AUC':>9} {'Med AUC':>9} {'Hard AUC':>9} {'Δ(E→H)':>9} {'退化率':>8}")
print("-" * 100)
for mn in auc_sorted:
    ea = all_results['Easy'][mn]['auc']
    ma = all_results['Medium'][mn]['auc']
    ha = all_results['Hard'][mn]['auc']
    delta = ha - ea
    deg_pct = abs(delta) / ea * 100 if ea > 0 else 0
    print(f"{mn:<20} {ea:>9.4f} {ma:>9.4f} {ha:>9.4f} {delta:>+9.4f} {deg_pct:>7.1f}%")
print("=" * 100)

# F1 汇总表 — 按 Hard F1 从大到小排序（None排最后）
f1_models = [m for m in all_models if all_results['Hard'][m]['f1'] is not None]
f1_none_models = [m for m in all_models if all_results['Hard'][m]['f1'] is None]
f1_sorted = sorted(f1_models, key=lambda m: all_results['Hard'][m]['f1'], reverse=True) + f1_none_models
print(f"\n{'Model':<20} {'Easy F1':>9} {'Med F1':>9} {'Hard F1':>9} {'Δ(E→H)':>9} {'退化率':>8}")
print("-" * 100)
for mn in f1_sorted:
    ef = all_results['Easy'][mn]['f1']
    mf = all_results['Medium'][mn]['f1']
    hf = all_results['Hard'][mn]['f1']
    if ef is None:
        print(f"{mn:<20} {'   -':>9} {'   -':>9} {'   -':>9} {'   -':>9} {'   -':>8}")
    else:
        delta = hf - ef
        deg_pct = abs(delta) / ef * 100 if ef > 0 else 0
        print(f"{mn:<20} {ef:>9.4f} {mf:>9.4f} {hf:>9.4f} {delta:>+9.4f} {deg_pct:>7.1f}%")
print("=" * 100)

total_elapsed = time.time() - t_total
print(f"\nTotal time: {total_elapsed/60:.1f} min")
print("Done!")
