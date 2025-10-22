# Preprocess
## Run this ONLY if:
##  - You changed RAW data (added/removed .mat, different split), or
##  - You changed rasterisation knobs that affect produced images:
##      * out_hw, channels, scale, step
##      * any logic inside rasterise_numpy.py (e.g., bilinear_splat, thresholding)
##  - You want cached NPZs for faster experiments.
##
## If you only changed model/training hyperparams (C1, C2, k, stride, padding,
## pool_size, lr, weight_decay, batch_size, epochs), you do NOT need to rerun this.
# PowerShell / CMD:
python -m src.scripts.preprocess_tactip_numpy --config configs/tactip_cnn.json --out_dir data/preprocessed

# Train model
## Trains from RAW .mat files using sequence-level split and on-the-fly rasterisation.
python -m src.scripts.train_tactip_numpy --config configs/tactip_cnn.json

# Evaluate
## Evaluates saved params on RAW .mat files, picks a threshold on VAL to maximise F1,
## then reports precision/recall/F1/acc on train-sub and test, and writes metrics JSON.
python -m src.scripts.eval_tactip_numpy --config configs/tactip_cnn.json --params configs/tactip_cnn_params.npz --out configs/tactip_metrics.json

# Diagnostics
python -m src.tests.diag_preproc --base_dir data/raw --split train --num_examples 5

