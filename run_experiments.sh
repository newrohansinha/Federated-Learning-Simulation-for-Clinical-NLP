python train_centralized.py --epochs 3 --batch_size 64 --lr 0.001 --dp 0 --results_path results/metrics.csv
python train_federated.py --num_clients 10 --num_rounds 5 --local_epochs 1 --batch_size 32 --lr 0.001 --attack 0 --dp 0 --defense none --results_path results/metrics.csv
python train_federated.py --num_clients 10 --num_rounds 5 --local_epochs 1 --batch_size 32 --lr 0.001 --attack 0 --dp 1 --epsilon 4.0 --defense none --results_path results/metrics.csv
python train_federated.py --num_clients 10 --num_rounds 5 --local_epochs 1 --batch_size 32 --lr 0.001 --attack 1 --dp 1 --epsilon 4.0 --defense none --results_path results/metrics.csv
python train_federated.py --num_clients 10 --num_rounds 5 --local_epochs 1 --batch_size 32 --lr 0.001 --attack 1 --dp 0 --defense clip --max_update_norm 1.0 --results_path results/metrics.csv
python report.py --results_path results/metrics.csv --out_path results/report.md
