import os
import torch
import argparse
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
import sys
sys.path.append("src")
from load_data import *
from config import Config

def draw_curve(args):
    
    data = args.data
    fold = args.fold

    if data.lower() == "mimic2":
        dataset = load_mimic2(fold=fold)
    elif data.lower() == "mimic3":
        dataset = load_mimic3(fold=fold)
    elif data.lower() == "income":
        dataset = load_income(fold=fold)
    elif data.lower() == "housing":
        dataset = load_housing()
    else:
        raise ValueError("Data {:s} not supported".format(args.data))


    problem = dataset['problem']
    X_train, y_train = dataset['X_train'], dataset['y_train']
    X_test, y_test = dataset['X_test'], dataset['y_test']
    X_val, y_val = dataset.get('X_val', None), dataset.get('y_val', None)

    if X_val is None:
        X = pd.concat([X_train, X_test]).reset_index(drop=True)
        y = np.concatenate([y_train, y_test])
    else:
        X = pd.concat([X_train, X_val, X_test]).reset_index(drop=True)
        y = np.concatenate([y_train, y_val, y_test])

    config = Config(
        data = args.data,
        model = args.model,
        lr = args.lr,
        max_epoch = args.max_epoch,
        batch_size = args.batch_size,
        test_step = args.test_step,
        p = args.p,
        h_dim = args.h_dim,
        n_proto = args.n_proto,
        n_layers = args.n_layers,
        n_layers_pred = args.n_layers_pred,
        tau = args.tau,
        batch_norm = False if args.batch_norm is None else True,
        dropout = args.dropout,
        dropout_output = args.dropout_output,
        output_penalty = args.output_penalty,
        weight_decay = args.weight_decay,
        eval = True, # evaluation mode
        exp_str = args.exp_str,
        device = args.device,
        fold = args.fold,
        seed = args.seed
    )

    checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pt")
    print("Loading checkpoint from:", checkpoint_path)
    config.model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
    config.model.eval()

    save_dir = config.checkpoint_dir.replace("checkpoint","fig")
    os.makedirs(save_dir, exist_ok=True)

    X_trans, _ = config.preprocessor.transform(X, y)
    cat_features = dataset.get("cat_features", [])
    if len(cat_features) > 0:
        cat2ord = {}
        ord2cat = {}
        X_ord = X.copy()
        for feat in cat_features:
            categories = sorted(X[feat].unique())
            cat2ord[feat] = {cat:i for i, cat in enumerate(categories)}
            ord2cat[feat] = {i:cat for i, cat in enumerate(categories)}
            X_ord[feat] = X_ord[feat].map(cat2ord[feat])
    else:
        X_ord = X

    loader = DataLoader(OurDataset(X_trans, y), batch_size = 512, shuffle = False)

    Logits_multi_layers = []
    with torch.no_grad():
        for x, _ in loader:
            Logits = []
            Z = config.model.encode(x.float().to(config.device), T=1e-8) # list of (batch_size, n_feat, h_dim)
            logits = 0
            for j in range(len(Z)):
                z_comp = Z[j][:,None] * config.model.mask[None,:,:,None] # (batch_size, n_comp, n_feat, h_dim)
                res = config.model.clfs[j](z_comp.flatten(start_dim=-2)) # (batch_size, n_comp, n_class)
                res = res * config.model.aggs[j].weight[0, None, :, None] / config.model.n_comp # (batch_size, n_comp, n_class)
                logits += res
                Logits.append(res)
            Logits_multi_layers.append(torch.stack(Logits).transpose(0,1)) # append (batch_size, n_layers, n_comp, n_class)
    Logits_multi_layers = torch.cat(Logits_multi_layers)
    if problem == "regression":
        Logits_multi_layers = Logits_multi_layers * config.preprocessor.y_std + config.preprocessor.y_mu
    Logits_multi_layers -= Logits_multi_layers.mean(dim=(0,1))[None,None]

    color = "blue"

    for i in tqdm.tqdm(range(config.model.n_feat)):
        unique_idx = np.unique(X.iloc[:, i], return_index=True)[1]

        logits = Logits_multi_layers[:, :, i, 0] # (dataset_size, n_layers)
        logits = logits[unique_idx, :].detach().cpu()

        if args.data == "mimic2":
            y_min = -1
            y_max = 2
        elif args.data == "income":
            y_min = -5
            y_max = 8
        else:
            y_min = min(-15, logits.min() - 0.5)
            y_max = max(15, logits.max() + 0.5)

        if  X.columns[i] == "Longitude":
            y_min = -5
            y_max = 5

        n_blocks = 20
        x_n_blocks = min(n_blocks, len(unique_idx))
        x_min = X.iloc[unique_idx,i].astype(float).min()
        x_max = X.iloc[unique_idx,i].astype(float).max()

        density = np.histogram(X.iloc[:, i].astype(float), bins=x_n_blocks)
        density = density[0] / np.max(density[0])

        length = (x_max - x_min) / x_n_blocks

        for j in range(args.n_layers):

            # plt.clf()
            plt.close()
            fig, ax = plt.subplots(figsize=(6,6))

            ax.plot(X_ord.iloc[unique_idx,i].astype(float), logits[:, j], label="logits", color=color, linewidth=3)
  
            for k in range(x_n_blocks):
                x_start = x_min + length * k
                x_end = x_start + length
                alpha = min(1.0, 0.01 + density[k])
                rect = patches.Rectangle(
                    (x_start, y_min),
                    x_end - x_start,
                    y_max - y_min,
                    linewidth=0.01,
                    edgecolor=[0.9, 0.5, 0.5],
                    facecolor=[0.9, 0.5, 0.5],
                    alpha=alpha,
                )
                ax.add_patch(rect)

            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            if X.columns[i] == "Population":
                plt.xticks([10000, 20000, 30000])
            if X.columns[i] == "CapitalLoss":
                plt.xticks([1000, 2000, 3000, 4000])
            plt.xticks(fontsize='x-large')
            plt.yticks(fontsize='x-large')
            plt.xlabel(X.columns[i], fontsize='x-large')
            plt.title("Layer " + str(j + 1), fontsize=18)
            plt.savefig(os.path.join(save_dir, X.columns[i]+"_"+str(j + 1)+".png"), bbox_inches='tight', dpi=1200)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data")
    parser.add_argument("--model")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_step", type=int, default=1)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--h_dim", type=int, default=64)
    parser.add_argument("--n_proto", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_layers_pred", type=int, default=2)
    parser.add_argument("--tau", type=float, default=16)
    parser.add_argument("--batch_norm", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dropout_output", type=float, default=0.0)
    parser.add_argument("--output_penalty", type=float, default=0.0)    
    parser.add_argument("--weight_decay", type=float, default=1e-8)
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction)
    parser.add_argument("--exp_str", type=str, help="special string to identify an experiment")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    
    assert args.p == 1
    
    draw_curve(args)