import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
import pandas as pd
import os
from conch.downstream.zeroshot_path import zero_shot_classifier, topj_pooling
from conch.open_clip_custom import create_model_from_pretrained

parser = argparse.ArgumentParser(description="CONCH multi-prompt Zero-shot classification")
parser.add_argument('--folder', type=str)
parser.add_argument('--project', type=str, default = "RCC")
args = parser.parse_args()

# Data preparation
if args.project == "RCC":
    idx_to_class = {0: "CHRCC", 1: "CCRCC", 2: "PRCC"}
    classes = ["Renal cell carcinoma, chromophobe type",
               "Clear cell adenocarcinoma, NOS",
               "Papillary adenocarcinoma, NOS"]
if args.project == "AI4SKIN":
    idx_to_class = {0: "lm", 1: "lms", 2: "df", 3: "dfs", 4: "mel", 5:"fxa"}
    classes = ["Leiomyoma",
               "Leiomyosarcoma",
               "Dermatofibroma",
               "Dermatofibrosarcoma",
               "Melanoma",
               "Fibroxhantoma"]
if args.project == "NSCLC":
    idx_to_class = {0: "LUAD", 1: "LUSC"}
    classes = ["Adenocarcinoma, NOS",
               "Squamous cell carcinoma, NOS"]

data = pd.read_csv("local_data/csv/" + args.project + ".csv")
features = [np.load(os.path.join(args.folder, fn + ".npy")) for fn in tqdm(data["WSI"])]
labels = np.array([classes.index(item) for item in data["GT"]], dtype=np.int64)

# Model building
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint_path = '../feature_extraction/local_data/model/pytorch_model.bin'
model, _ = create_model_from_pretrained(model_cfg='conch_ViT-B-16', checkpoint_path=checkpoint_path, device=device)
_ = model.eval()

# Prompt ensembling
BACC_mean, BACC_top1, BACC_top5, BACC_top10, BACC_top50, BACC_top100, BACC_top500, BACC_top1000 = [], [], [], [], [], [], [], []
prompt_file = './prompts/' + args.project + '.json'
with open(prompt_file) as f:
    all_prompts = json.load(f)

for it in tqdm(range(len(all_prompts))):
    prompts = all_prompts[str(it)]
    classnames = prompts['classnames']
    templates = prompts['templates']
    n_classes = len(classnames)
    classnames_text = [classnames[str(idx_to_class[idx])] for idx in range(n_classes)]
    zeroshot_weights = zero_shot_classifier(model, classnames_text, templates, device=device)

    # zeroshot_weights = np.load("../MILAdapters/local_data/prompts/NSCLC_promptensemble.npy")
    # zeroshot_weights = torch.tensor(zeroshot_weights).T.cuda()
    preds_top1, preds_top5, preds_top10, preds_top50, preds_top100, preds_top500, preds_top1000, preds_mean = [], [], [], [], [], [], [], []
    for image_features in features:
        image_features = torch.tensor(image_features).cuda()
        image_features = model.visual.forward_project(image_features)
        image_features = F.normalize(image_features, dim=-1)
        logits = image_features @ zeroshot_weights
        preds, _ = topj_pooling(logits, topj = (1,5,10,50,100,500,1000))

        preds_top1.append(preds[1].item())
        preds_top5.append(preds[5].item())
        preds_top10.append(preds[10].item())
        preds_top50.append(preds[50].item())
        preds_top100.append(preds[100].item())
        preds_top500.append(preds[500].item())
        preds_top1000.append(preds[1000].item())

        preds = np.argmax((image_features.mean(dim=0)@zeroshot_weights).detach().cpu().numpy())
        preds_mean.append(preds.item())

    BACC_mean.append(balanced_accuracy_score(labels, preds_mean))
    BACC_top1.append(balanced_accuracy_score(labels, preds_top1))
    BACC_top5.append(balanced_accuracy_score(labels, preds_top5))
    BACC_top10.append(balanced_accuracy_score(labels, preds_top10))
    BACC_top50.append(balanced_accuracy_score(labels, preds_top50))
    BACC_top100.append(balanced_accuracy_score(labels, preds_top100))
    BACC_top500.append(balanced_accuracy_score(labels, preds_top500))
    BACC_top1000.append(balanced_accuracy_score(labels, preds_top1000))

df = pd.DataFrame(
    {'BACC_mean': BACC_mean,
     'BACC_top1': BACC_top1,
     'BACC_top5': BACC_top5,
     'BACC_top10': BACC_top10,
     'BACC_top50': BACC_top50,
     'BACC_top100': BACC_top100,
     'BACC_top500': BACC_top500,
     'BACC_top1000': BACC_top1000,
     })
df.to_excel(args.project + '_MIZero.xlsx', index=False)  # Set index=False if you don't want the DataFrame index in the file