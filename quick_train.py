"""快速训练 - 减少epoch用于演示"""
import os, json, pickle, copy, io, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DATA_DIR = "data"; MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32; EPOCHS = 30; PATIENCE = 8
EMBED_DIM = 300; NUM_FILTERS = 200; KERNEL_SIZES = [2,3,4,5]
LSTM_HIDDEN = 256; LSTM_LAYERS = 2; DROPOUT = 0.35
WEIGHT_DECAY = 2e-4; LR = 5e-4
FOCAL_GAMMA = 2.0; LABEL_SMOOTHING = 0.08

torch.manual_seed(42); np.random.seed(42)

vocab = pickle.load(open(os.path.join(DATA_DIR, "vocab.pkl"), "rb"))
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy")).ravel()
X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
y_val = np.load(os.path.join(DATA_DIR, "y_val.npy")).ravel()
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy")).ravel()

VOCAB_SIZE = len(vocab); NUM_CLASSES = 3

print(f"Vocab: {VOCAB_SIZE}, Device: {DEVICE}")
print(f"Train: {len(X_train)} ({np.bincount(y_train)})")
print(f"Val: {len(X_val)} ({np.bincount(y_val)})")
print(f"Test: {len(X_test)} ({np.bincount(y_test)})")

from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_train)
cw_tensor = torch.tensor(cw, dtype=torch.float, device=DEVICE)
print(f"Class weights: {cw}")

class Model(nn.Module):
    def __init__(self, vs, ed=300, nf=200, ks=[2,3,4,5], lh=256, ll=2, nc=3, do=0.35):
        super().__init__()
        self.emb = nn.Embedding(vs, ed, padding_idx=0)
        self.edrop = nn.Dropout(do*0.6)
        self.convs = nn.ModuleList([nn.Conv1d(ed, nf, k, padding=k//2) for k in ks])
        self.bns = nn.ModuleList([nn.BatchNorm1d(nf) for _ in ks])
        self.cdrop = nn.Dropout(do*0.6)
        tf = nf*len(ks)
        self.lstm = nn.LSTM(tf, lh, ll, batch_first=True, bidirectional=True, dropout=do if ll>1 else 0)
        self.ldrop = nn.Dropout(do)
        lo = lh*2
        self.attn = nn.MultiheadAttention(lo, 4, dropout=do*0.5, batch_first=True)
        self.anorm = nn.LayerNorm(lo)
        self.fc1 = nn.Linear(lo, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fdrop = nn.Dropout(do)
        self.fc2 = nn.Linear(128, nc)
    def forward(self, x):
        B,L=x.shape; e=self.edrop(self.emb(x)).permute(0,2,1)
        cs=[F.relu(bn(F.adaptive_max_pool1d(c(e),L))) for c,bn in zip(self.convs,self.bns)]
        cc=self.cdrop(torch.cat(cs,1)).permute(0,2,1)
        lo,_=self.lstm(cc); lo=self.ldrop(lo)
        ao,_=self.attn(lo,lo,lo); lo=self.anorm(lo+ao)
        p=F.adaptive_max_pool1d(lo.permute(0,2,1),1).squeeze(-1)
        return self.fc2(self.fdrop(F.relu(self.bn1(self.fc1(p)))))

class FL(nn.Module):
    def __init__(self, a=None, g=2.0, ls=0.0):
        super().__init__(); self.a=a; self.g=g; self.ls=ls
    def forward(self, logits, tgt):
        nc=logits.size(-1)
        if self.ls>0:
            oh=F.one_hot(tgt,nc).float()*(1-self.ls)+self.ls/nc
            ce=-(oh*F.log_softmax(logits,-1)).sum(-1)
            pt=(torch.softmax(logits,-1)*oh).sum(-1)
        else:
            ce=F.cross_entropy(logits,tgt,reduction='none'); pt=torch.exp(-ce)
        fw=(1-pt)**self.g
        if self.a is not None: fw=fw*self.a[tgt]
        return (fw*ce).mean()

def mkld(X,y,s=True):
    ds=TensorDataset(torch.tensor(X,dtype=torch.long),torch.tensor(y,dtype=torch.long))
    return DataLoader(ds,BATCH_SIZE,shuffle=s)

tl=mkld(X_train,y_train,True); vl=mkld(X_val,y_val,False); tel=mkld(X_test,y_test,False)

model=Model(VOCAB_SIZE,EMBED_DIM,NUM_FILTERS,KERNEL_SIZES,LSTM_HIDDEN,LSTM_LAYERS,NUM_CLASSES,DROPOUT).to(DEVICE)
np_=sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: {np_:,}")

crit=FL(cw_tensor,FOCAL_GAMMA,LABEL_SMOOTHING)
opt=torch.optim.AdamW(model.parameters(),LR,weight_decay=WEIGHT_DECAY)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-6)

bv=0.0; bs=None; be=0; pc=0; hist={"tl":[],"vl":[],"va":[]}
print("Training...")
for ep in range(1,EPOCHS+1):
    model.train(); tl_=0.0
    for xb,yb in tl:
        xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad()
        L=crit(model(xb),yb); L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tl_+=L.item()*xb.size(0)
    tl_/=len(tl.dataset)
    model.eval(); vl_=0.0; vp,vt=[],[]
    with torch.no_grad():
        for xb,yb in vl:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); lo=model(xb)
            vl_+=crit(lo,yb).item()*xb.size(0)
            vp.extend(lo.argmax(1).cpu().numpy()); vt.extend(yb.cpu().numpy())
    vl_/=len(vl.dataset); va=accuracy_score(vt,vp); sch.step()
    hist["tl"].append(round(tl_,6)); hist["vl"].append(round(vl_,6)); hist["va"].append(round(float(va),6))
    s=""
    if va>bv: bv=va; bs=copy.deepcopy(model.state_dict()); be=ep; pc=0; s=" *"
    else: pc+=1
    print(f"E{ep:3d} | TL:{tl_:.4f} | VL:{vl_:.4f} | VA:{va:.4f}{s}")
    if pc>=PATIENCE: print(f"Early stop@{ep}"); break

model.load_state_dict(bs); model.eval()
tp,tt=[],[]
with torch.no_grad():
    for xb,yb in tel:
        xb,yb=xb.to(DEVICE),yb.to(DEVICE)
        lo=model(xb); tp.extend(lo.argmax(1).cpu().numpy()); tt.extend(yb.cpu().numpy())
ta=accuracy_score(tt,tp)
cm_=confusion_matrix(tt,tp)
print(f"\nBest Ep:{be}, VA:{bv:.4f}, TA:{ta:.4f}")
print(classification_report(tt,tp,target_names=['负面','中性','正面'],digits=4))
print("CM:")
print(cm_)
for i,n in enumerate(['负面','中性','正面']):
    rs=cm_[i].sum(); print(f"  {n}: {cm_[i][i]/rs:.4f} ({cm_[i][i]}/{rs})")

torch.save({"model_state":bs,"model_type":"TextCNN_BiLSTM_Attention","vocab_size":VOCAB_SIZE,"embed_dim":EMBED_DIM,"num_filters":NUM_FILTERS,"kernel_sizes":KERNEL_SIZES,"lstm_hidden":LSTM_HIDDEN,"lstm_layers":LSTM_LAYERS,"num_classes":NUM_CLASSES,"dropout":DROPOUT,"test_acc":float(ta),"best_val_acc":float(bv),"best_epoch":be},os.path.join(MODEL_DIR,"model_export.pt"))
cfg={"model_type":"TextCNN_BiLSTM_Attention","vocab_size":VOCAB_SIZE,"embed_dim":EMBED_DIM,"num_filters":NUM_FILTERS,"kernel_sizes":KERNEL_SIZES,"lstm_hidden":LSTM_HIDDEN,"lstm_layers":LSTM_LAYERS,"num_classes":NUM_CLASSES,"dropout":DROPOUT,"max_seq_len":64,"label_map":{"0":"负面","1":"中性","2":"正面"},"test_acc":float(ta),"best_val_acc":float(bv)}
with open(os.path.join(MODEL_DIR,"model_config.json"),"w",encoding="utf-8") as f: json.dump(cfg,f,indent=2,ensure_ascii=False)
metrics={"model_type":"TextCNN_BiLSTM_Attention","test_accuracy":float(ta),"best_val_accuracy":float(bv),"best_epoch":be,"num_params":np_,"history":hist,"classification_report":classification_report(tt,tp,target_names=['负面','中性','正面'],output_dict=True),"confusion_matrix":cm_.tolist()}
with open(os.path.join(MODEL_DIR,"metrics.json"),"w",encoding="utf-8") as f: json.dump(metrics,f,indent=2,ensure_ascii=False)
print(f"\nModel saved to {MODEL_DIR}/model_export.pt")
print(f"Config saved to {MODEL_DIR}/model_config.json")
print("Done!")