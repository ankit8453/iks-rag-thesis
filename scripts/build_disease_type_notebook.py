"""Generate notebooks/disease_type_train.ipynb — the ISOLATED disease-type experiment.

Trains a crop-agnostic disease-TYPE classifier and pushes it to a NEW HF repo
(`iks-disease-type-v1`). It never touches the current C-PD model, its repos, the
existing datasets, or app/config — so if it is worse, we simply do not deploy it
and the current system is unchanged (see the memory rule).

Leaf-focus lesson carried over: the model is warm-started from the leaf-attentive
PlantVillage backbone, and a Grad-CAM sanity cell checks it looks at the leaf.
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
# Disease-type experiment — crop-agnostic classifier (ISOLATED)

Trains a model that predicts the **disease type** (rust, leaf_spot, blight,
mildew, …) pooled across crops, instead of crop-specific classes. This is a
research experiment to see if it beats the current crop-specific model.

**Safety (important):** this touches **nothing** in the current system. It builds
a new dataset folder and pushes the model to a **new** repo
`ankit-iiitdmj/iks-disease-type-v1`. The deployed C-PD model
(`iks-disease-plantdoc-crop`), its sibling repos, `data/splits/*`, and
`app/config.py` are left exactly as they are. If this model is worse, do nothing.

**Requirements:** T4 GPU runtime + HF write token. You will **upload the
Brazilian dataset zip** (`05-11-2020_...zip`) in Cell 3.
"""),

    md("## Cell 1 — clone + install + GPU/token check"),
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
from huggingface_hub import login; login()
import torch
assert torch.cuda.is_available(), "Switch Runtime -> T4 GPU"
print("GPU:", torch.cuda.get_device_name(0))"""),

    md("""\
## Cell 2 — get the source datasets

Downloads PlantVillage / PlantDoc / Paddy (the same ones the current model used).
This is a few GB and takes a while."""),
    code("""\
for s in ["download_plantvillage.py","download_plantdoc.py","download_paddy_doctor.py"]:
    print("running", s, "...")
    subprocess.run([sys.executable, f"scripts/{s}"], check=False)

import glob
def find_root(*cands):
    for c in cands:
        if os.path.isdir(c) and glob.glob(c+"/**/*.jpg", recursive=True): return c
    return None
PV=find_root("data/plant_disease/plantvillage/raw")
PD=find_root("data/plant_disease/plantdoc/raw")
PA=find_root("data/plant_disease/paddy_doctor/raw","data/plant_disease/paddy/raw")
print("PlantVillage:",PV,"\\nPlantDoc:",PD,"\\nPaddy:",PA)"""),

    md("""\
## Cell 3 — upload the Brazilian dataset, then BUILD the disease-type dataset

Run the cell, click **Choose Files**, pick `05-11-2020_...zip`. It unzips, then
the prep script unifies everything into `data/disease_type/{train,val,test}` by
disease type (deduped, resized to 384)."""),
    code("""\
from google.colab import files
up = files.upload()                      # pick the 05-11-2020_...zip
import zipfile
zname = next(k for k in up if k.lower().endswith(".zip"))
with zipfile.ZipFile(zname) as z: z.extractall("/content/brazil")
BR="/content/brazil"

srcs=[]
if PV: srcs += ["--source", f"plantvillage={PV}"]
if PD: srcs += ["--source", f"plantdoc={PD}"]
if PA: srcs += ["--source", f"paddy={PA}"]
srcs += ["--source", f"brazil={BR}"]

subprocess.run([sys.executable,"scripts/build_disease_type_dataset.py",
                *srcs,"--out","data/disease_type","--size","384","--min-per-class","60"],
               check=True)"""),

    md("## Cell 4 — train (warm-started from the leaf-attentive PlantVillage backbone)"),
    code("""\
import torch, timm, json, time
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.disease.train_crop import build_crop_model
from src.disease.train import CheckpointManager

SIZE=380; DEV="cuda"
mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]
train_tf=transforms.Compose([transforms.RandomResizedCrop(SIZE,scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(), transforms.Normalize(mean,std)])
eval_tf=transforms.Compose([transforms.Resize(int(SIZE*1.05)), transforms.CenterCrop(SIZE),
    transforms.ToTensor(), transforms.Normalize(mean,std)])

train_ds=datasets.ImageFolder("data/disease_type/train", train_tf)
val_ds  =datasets.ImageFolder("data/disease_type/val",   eval_tf)
CLASSES=train_ds.classes; N=len(CLASSES)
print("classes:",CLASSES)
tl=DataLoader(train_ds,batch_size=16,shuffle=True,num_workers=2,pin_memory=True)
vl=DataLoader(val_ds,batch_size=16,shuffle=False,num_workers=2,pin_memory=True)

model=build_crop_model(num_classes=N,
    backbone_repo="ankit-iiitdmj/iks-disease-plantvillage").to(DEV)  # leaf-attentive warm start
opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
crit=nn.CrossEntropyLoss(label_smoothing=0.1)
scaler=torch.cuda.amp.GradScaler()
ckpt=CheckpointManager("ankit-iiitdmj/iks-disease-type-v1")   # NEW repo — nothing else touched

def acc(loader):
    model.eval(); ok=tot=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(DEV),y.to(DEV)
            ok+=(model(x).argmax(1)==y).sum().item(); tot+=y.numel()
    return ok/tot

EPOCHS=15; best=0.0; hist=[]
for ep in range(EPOCHS):
    model.train(); t=time.time()
    for x,y in tl:
        x,y=x.to(DEV),y.to(DEV); opt.zero_grad()
        with torch.cuda.amp.autocast():
            loss=crit(model(x),y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    va=acc(vl); hist.append({"epoch":ep,"val_acc":va})
    is_best=va>best; best=max(best,va)
    print(f"epoch {ep+1}/{EPOCHS}  val_acc={va:.4f}  best={best:.4f}  ({time.time()-t:.0f}s)")
    try:
        ckpt.save_epoch(model,opt,None,ep,best,hist,is_best)   # pushes to the NEW repo
    except Exception as e:
        print("checkpoint push skipped:",e)
json.dump({"classes":CLASSES}, open("data/disease_type/classes.json","w"))
print("best val acc:",best)"""),

    md("## Cell 5 — test accuracy + per-class report"),
    code("""\
from sklearn.metrics import classification_report, confusion_matrix
test_ds=datasets.ImageFolder("data/disease_type/test", eval_tf)
te=DataLoader(test_ds,batch_size=16,shuffle=False,num_workers=2)
model.eval(); yp=[]; yt=[]
with torch.no_grad():
    for x,y in te:
        yp += model(x.to(DEV)).argmax(1).cpu().tolist(); yt += y.tolist()
print("TEST accuracy:", sum(int(a==b) for a,b in zip(yp,yt))/len(yt))
print(classification_report(yt,yp,target_names=test_ds.classes,digits=3))"""),

    md("""\
## Cell 6 — leaf-focus sanity (Grad-CAM) + how to read this

The current C-PD model's whole point was attending to the leaf. Check the new
model does too, so we don't re-introduce background bias."""),
    code("""\
import numpy as np, matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
try:
    target_layer = model._backbone.blocks[-2]        # timm efficientnet_b4
except Exception:
    target_layer = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)][-1]
cam=GradCAM(model=model, target_layers=[target_layer])

xb,yb=next(iter(te))
fig,ax=plt.subplots(1,4,figsize=(14,4))
for i in range(4):
    g=cam(input_tensor=xb[i:i+1].to(DEV))[0]
    img=(xb[i].permute(1,2,0).numpy()*std+mean).clip(0,1)
    ax[i].imshow(show_cam_on_image(img.astype(np.float32), g, use_rgb=True))
    ax[i].set_title(test_ds.classes[yb[i]]); ax[i].axis("off")
plt.show()"""),

    md("""\
### Reading the result — honestly

- **Disease-type test accuracy is NOT directly comparable to the current model's
  66.6%** — different label space (fewer, coarser classes). Expect it higher
  *because* the task is easier; that alone is not "better".
- What would make this worth deploying: **(a)** high per-class accuracy including
  on the pooled classes, **(b)** Grad-CAM still on the leaf (above), and **(c)**
  it generalises across crops (a rust image from a crop it never trained on is
  still called rust).
- The model is in `iks-disease-type-v1`. The current system is untouched.
  Deploying would be a deliberate, reversible `DISEASE_MODEL_REPO` switch — only
  if the evidence says it's genuinely better.
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
