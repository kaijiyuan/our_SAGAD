# Mitigating Homophily Disparity in Graph Anomaly Detection

Official implementation of "Mitigating Homophily Disparity in Graph Anomaly Detection: A Scalable and Adaptive Approach".

## Environment

The code is written for Python and PyTorch/DGL. The expected dependency set is:

```text
python 3.10
torch 2.4.0
dgl 2.4.0+cu124
numpy 2.1.3
scipy 1.14.1
tqdm
pyyaml
scikit-learn
```

On the current Windows workstation, the existing environment below has also passed a reddit smoke test:

```powershell
E:\FISF\.conda-fisf\python.exe --version
```

Observed versions in that environment:

```text
python 3.10.20
torch 2.3.0+cu121
dgl 2.2.1
```

If you use this local environment, run commands with the full Python path:

```powershell
& 'E:\FISF\.conda-fisf\python.exe' main.py --dataset reddit
```

## Data

Datasets are loaded from `all_data/<dataset>`. The repository currently expects files such as:

```text
all_data/reddit
all_data/weibo
all_data/yelp
all_data/tolokers
all_data/questions
```

The data format follows the graph files provided by [GADBench](https://github.com/squareRoot3/GADBench). Each graph should contain node fields for `feature`, `label`, `train_masks`, `val_masks`, and `test_masks`.

Additional datasets can be downloaded from the [GADBench dataset link](https://drive.google.com/file/d/1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1/view?usp=sharing). DGraph-Fin and Elliptic must be downloaded separately from their original sources because of their licenses.

## Configuration

Training hyperparameters are defined in `semi_train.conf.yaml`. The `default` section is shared, and each dataset section can override it.

Common options:

```text
epochs        Maximum training epochs
patience      Early stopping patience
K             Chebyshev polynomial order
hid_dims      MLP hidden dimensions
feat_trans    Feature preprocessing mode
fusion        Selected from command-line arguments
```

Command-line arguments are defined in `main.py`. Useful options include:

```powershell
& 'E:\FISF\.conda-fisf\python.exe' main.py --dataset reddit --device 0 --fusion NodeDimGated --exp_setting tran
```

If CUDA is not available, the code automatically falls back to CPU.

## Smoke Test

A quick reddit smoke test can be used to verify the environment, data loading, MRQ sampling, feature masking, Chebyshev precomputation, and one model forward pass without running the full training loop:

```powershell
@'
import torch
import dgl
from dataloader import load_data
from mrqsampler import mrqsample
from feature_missing import apply_missing_features, impute_missing_features
from model import SAGAD, precompute_Tx, NodeDimGatedFusion

print('torch', torch.__version__)
print('dgl', dgl.__version__)

G, X, Y, train_masks, val_masks, test_masks = load_data('reddit', feat_trans='no')
print('loaded', G.num_nodes(), G.num_edges(), tuple(X.shape))

G_mrq = mrqsample('reddit', G, X, one_hop=True)
X_masked = impute_missing_features(apply_missing_features(X, 0.995, seed=0), method='zero')
Tx_list = precompute_Tx(G, X_masked, K=1, conv_norm='none')
X_nei = dgl.ops.copy_u_mean(G_mrq, X_masked)
X_final = torch.concat([X_masked, X_nei], dim=1)

in_dim = X_final.shape[1]
model = SAGAD(
    in_dim,
    in_dim // 2,
    hid_dims=[32, 2],
    K=1,
    dropout=0.0,
    activation='ReLU',
    mlp_norm='none',
    fusion=NodeDimGatedFusion(in_dim, in_dim // 2, 1),
)

idx = torch.arange(min(128, X_final.shape[0]))
with torch.no_grad():
    logits, coefs = model([t[idx] for t in Tx_list], X_final[idx], return_coefs=True)

print('forward', tuple(logits.shape), tuple(coefs.shape), bool(torch.isfinite(logits).all()))
print('SMOKE_OK')
'@ | & 'E:\FISF\.conda-fisf\python.exe' -
```

Expected final output includes:

```text
SMOKE_OK
```

## Training And Evaluation

Run full training on reddit:

```powershell
& 'E:\FISF\.conda-fisf\python.exe' main.py --dataset reddit
```

Run with the active Python environment instead:

```bash
python main.py --dataset reddit
```

Supported dataset names are:

```text
reddit, weibo, amazon, yelp, tfinance, elliptic, tolokers, questions, dgraphfin, tsocial
```

Training writes logs to `logs/<dataset>.log` and model snapshots to `snapshots/<dataset>_snapshot.pkl`.

## Acknowledgements

The code is implemented based on [GADBench](https://github.com/squareRoot3/GADBench), [UniGAD](https://github.com/lllyyq1121/UniGAD), and [PolyGCL](https://github.com/ChenJY-Count/PolyGCL).
