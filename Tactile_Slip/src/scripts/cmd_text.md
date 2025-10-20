# src/scripts/cmd_test.py 

## Preprocess
python -m src.scripts.preprocess_tactip_numpy --config configs/tactip_cnn.json --out_dir data/preprocessed

## Train
python -m src.scripts.train_tactip_numpy --config configs/tactip_cnn.json

## Test
python -m src.scripts.eval_tactip_numpy --config configs/tactip_cnn.json --params configs/tactip_cnn_params.npz --out configs/tactip_metrics.json
