"""Generate notebooks/disease_type_train.ipynb — the ISOLATED disease-type experiment.

HF-native: no manual Colab/Drive uploads. The raw Brazilian set lives in a private
HF dataset repo; the built, unified disease-type set is pushed to another HF
dataset repo and pulled on later runs. The trained model goes to a NEW model repo
`iks-disease-type-v1`. Nothing here touches the current C-PD model, its repos,
data/splits, or app/config — if it is worse, deploy nothing.

Leaf-focus lesson carried over: warm-started from the leaf-attentive PlantVillage
backbone + a Grad-CAM sanity cell.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "disease_type_train.ipynb"


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}


def code(s: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": s.splitlines(True)}


CELLS = [
    md("""\
# Disease-type experiment — crop-agnostic classifier (ISOLATED, HF-native)

Predicts the **disease type** (rust, leaf_spot, blight, mildew, …) pooled across
crops. Research experiment to see if it beats the current crop-specific model.

**No uploads.** All data lives on Hugging Face:
- raw Brazilian set → `ankit-iiitdmj/iks-brazil-multicrop` (push it once — see the
  one-line helper Ankit was given)
- built, unified disease-type set → `ankit-iiitdmj/iks-disease-type-data`
- trained model → `ankit-iiitdmj/iks-disease-type-v1`

**Safety:** touches nothing in the current system. The deployed C-PD model
(`iks-disease-plantdoc-crop`), its sibling repos, `data/splits/*`, and
`app/config.py` are untouched. If this is worse, do nothing.

**Requirements:** T4 GPU runtime + HF write token.
"""),

    md("## Cell 1 — clone + install + GPU/token"),
    code("""\
import os, shutil, subprocess, sys
REPO="/content/iks-rag-thesis"; URL="https://github.com/ankit8453/iks-rag-thesis.git"
os.chdir("/content"); shutil.rmtree(REPO, ignore_errors=True)
env=os.environ.copy(); env["GIT_LFS_SKIP_SMUDGE"]="1"
r=subprocess.run(["git","clone",URL,REPO],env=env,capture_output=True,text=True)
if r.returncode: print(r.stderr); raise RuntimeError("clone failed")
os.chdir(REPO); sys.path.insert(0,REPO)

subprocess.run([sys.executable,"-m","pip","install","-q","timm>=1.0","huggingface_hub>=0.24",
                "datasets>=2.20","pillow","scikit-learn","grad-cam>=1.5"],check=True)

# HF login — same as the other notebooks: paste your write token in the box.
from huggingface_hub import HfApi, login
login()
print("HF user:", HfApi().whoami().get("name"))

import torch
assert torch.cuda.is_available(), "Switch Runtime -> T4 GPU"
print("GPU:", torch.cuda.get_device_name(0))

BRAZIL_REPO = "ankit-iiitdmj/iks-brazil-multicrop"   # raw Brazilian zip (push once)
DATA_REPO   = "ankit-iiitdmj/iks-disease-type-data"  # built, unified set
MODEL_REPO  = "ankit-iiitdmj/iks-disease-type-v1"    # trained model (NEW)"""),

    md("""\
## Cell 2 — get the disease-type dataset into `data/disease_type/`

**Fast path:** if the unified set already exists at `DATA_REPO`, it is pulled
from HF and materialised to folders (skips the multi-GB raw downloads).

**Build path** (first run, `BUILD_FROM_RAW=True`): downloads PlantVillage /
PlantDoc / Paddy, pulls the Brazilian zip from HF, unifies by disease type, then
pushes the result to `DATA_REPO` so later runs use the fast path."""),
    code("""\
import glob
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download, HfApi

BUILD_FROM_RAW = False    # set True the first time (or to rebuild from sources)
OUTDIR = "data/disease_type"

def materialise(dsdict):
    for split in dsdict:
        feat = dsdict[split].features["label"]
        for i, ex in enumerate(dsdict[split]):
            lab = feat.int2str(ex["label"])
            d = Path(OUTDIR)/split/lab; d.mkdir(parents=True, exist_ok=True)
            ex["image"].convert("RGB").save(d/f"{i:06d}.jpg", quality=90)

built = False
if not BUILD_FROM_RAW:
    try:
        materialise(load_dataset(DATA_REPO))
        built = True
        print("pulled prebuilt disease-type set from", DATA_REPO)
    except Exception as e:
        print("no prebuilt set yet (", e, ") -> building from raw"); BUILD_FROM_RAW = True

if BUILD_FROM_RAW and not built:
    for s in ["download_plantvillage.py","download_plantdoc.py","download_paddy_doctor.py"]:
        print("running", s, "..."); subprocess.run([sys.executable, f"scripts/{s}"], check=False)
    def find_root(*cs):
        for c in cs:
            if os.path.isdir(c) and glob.glob(c+"/**/*.jpg", recursive=True): return c
        return None
    PV=find_root("data/plant_disease/plantvillage/raw")
    PD=find_root("data/plant_disease/plantdoc/raw")
    PA=find_root("data/plant_disease/paddy_doctor/raw","data/plant_disease/paddy/raw")

    zpath = hf_hub_download(BRAZIL_REPO, "brazil.zip", repo_type="dataset")  # from HF, no upload
    import zipfile; zipfile.ZipFile(zpath).extractall("/content/brazil")

    srcs=[]
    for name,root in [("plantvillage",PV),("plantdoc",PD),("paddy",PA),("brazil","/content/brazil")]:
        if root: srcs += ["--source", f"{name}={root}"]
    subprocess.run([sys.executable,"scripts/build_disease_type_dataset.py",*srcs,
                    "--out",OUTDIR,"--size","384","--min-per-class","60"],check=True)

    # push the built set to HF so future runs skip the raw download + build
    try:
        load_dataset("imagefolder", data_dir=OUTDIR).push_to_hub(DATA_REPO, private=True)
        print("pushed unified set ->", DATA_REPO)
    except Exception as e:
        print("push skipped:", e)

print("classes:", sorted(os.listdir(f"{OUTDIR}/train")))"""),

    md("## Cell 3 — train (warm-started from the leaf-attentive PlantVillage backbone)"),
    code("""\
import torch, time, json
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.disease.train_crop import build_crop_model
from src.disease.train import CheckpointManager

SIZE=380; DEV="cuda"; mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]
train_tf=transforms.Compose([transforms.RandomResizedCrop(SIZE,scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(), transforms.Normalize(mean,std)])
eval_tf=transforms.Compose([transforms.Resize(int(SIZE*1.05)), transforms.CenterCrop(SIZE),
    transforms.ToTensor(), transforms.Normalize(mean,std)])

train_ds=datasets.ImageFolder("data/disease_type/train", train_tf)
val_ds  =datasets.ImageFolder("data/disease_type/val",   eval_tf)
CLASSES=train_ds.classes; N=len(CLASSES); print("classes:",CLASSES)
tl=DataLoader(train_ds,batch_size=16,shuffle=True,num_workers=2,pin_memory=True)
vl=DataLoader(val_ds,batch_size=16,shuffle=False,num_workers=2,pin_memory=True)

model=build_crop_model(num_classes=N,
    backbone_repo="ankit-iiitdmj/iks-disease-plantvillage").to(DEV)   # leaf-attentive warm start
opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
crit=nn.CrossEntropyLoss(label_smoothing=0.1); scaler=torch.cuda.amp.GradScaler()
ckpt=CheckpointManager(MODEL_REPO)                                    # NEW repo

def val_acc():
    model.eval(); ok=tot=0
    with torch.no_grad():
        for x,y in vl:
            x,y=x.to(DEV),y.to(DEV); ok+=(model(x).argmax(1)==y).sum().item(); tot+=y.numel()
    return ok/tot

EPOCHS=15; best=0.0; hist=[]
for ep in range(EPOCHS):
    model.train(); t=time.time()
    for x,y in tl:
        x,y=x.to(DEV),y.to(DEV); opt.zero_grad()
        with torch.cuda.amp.autocast(): loss=crit(model(x),y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    va=val_acc(); hist.append({"epoch":ep,"val_acc":va}); is_best=va>best; best=max(best,va)
    print(f"epoch {ep+1}/{EPOCHS}  val_acc={va:.4f}  best={best:.4f}  ({time.time()-t:.0f}s)")
    try: ckpt.save_epoch(model,opt,None,ep,best,hist,is_best)
    except Exception as e: print("checkpoint push skipped:",e)
json.dump({"classes":CLASSES}, open("data/disease_type/classes.json","w"))
print("best val acc:",best)"""),

    md("## Cell 4 — test accuracy + per-class report"),
    code("""\
from sklearn.metrics import classification_report
test_ds=datasets.ImageFolder("data/disease_type/test", eval_tf)
te=DataLoader(test_ds,batch_size=16,shuffle=False,num_workers=2)
model.eval(); yp=[]; yt=[]
with torch.no_grad():
    for x,y in te:
        yp += model(x.to(DEV)).argmax(1).cpu().tolist(); yt += y.tolist()
print("TEST accuracy:", sum(int(a==b) for a,b in zip(yp,yt))/len(yt))
print(classification_report(yt,yp,target_names=test_ds.classes,digits=3))"""),

    md("## Cell 5 — leaf-focus sanity (Grad-CAM)"),
    code("""\
import numpy as np, matplotlib.pyplot as plt, random
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
# DiseaseClassifier is a delegation wrapper; Grad-CAM needs the real nn.Module.
gm = model._module
try: target=model._backbone.blocks[-2]
except Exception: target=[m for m in gm.modules() if isinstance(m,torch.nn.Conv2d)][-1]
cam=GradCAM(model=gm, target_layers=[target])

# ORIGINAL next to Grad-CAM, across DIVERSE classes (the test loader is sorted,
# so a plain batch is all one class), with true vs predicted labels — so you can
# judge whether the hot region sits on the actual lesion.
idxs=random.sample(range(len(test_ds)), 6)
fig,ax=plt.subplots(len(idxs),2,figsize=(7,3.2*len(idxs))); model.eval()
for r,idx in enumerate(idxs):
    x,y=test_ds[idx]; xb1=x.unsqueeze(0).to(DEV)
    with torch.no_grad(): pred=int(model(xb1).argmax(1).item())
    g=cam(input_tensor=xb1)[0]
    img=(x.permute(1,2,0).numpy()*std+mean).clip(0,1).astype(np.float32)
    ax[r,0].imshow(img); ax[r,0].axis("off"); ax[r,0].set_title(f"original — true: {test_ds.classes[y]}")
    ax[r,1].imshow(show_cam_on_image(img,g,use_rgb=True)); ax[r,1].axis("off")
    ax[r,1].set_title(f"Grad-CAM — pred: {test_ds.classes[pred]} ({'correct' if pred==y else 'WRONG'})")
plt.tight_layout(); plt.show()"""),

    md("""\
### Reading the result — honestly

- Disease-type accuracy is **NOT** directly comparable to the current model's
  66.6% — different, coarser label space. A higher number alone is not "better".
- Worth deploying only if: **(a)** solid per-class accuracy, **(b)** Grad-CAM still
  on the leaf (above), **(c)** it generalises across crops. The model lives in
  `iks-disease-type-v1`; the current system is untouched; deploying is a
  deliberate, reversible `DISEASE_MODEL_REPO` switch.
"""),
]


def main() -> int:
    nb = {"cells": CELLS,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
