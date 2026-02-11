# QuadraFormer: Unified Query and Resource Forecasting for Database Workloads

This repository provides the official implementation of the **QuadraFormer** model, proposed in our ICDM 2025 paper. QuadraFormer is designed to jointly forecast SQL query parameters and system-level resource usage (e.g., CPU, memory, QPS) using a unified, multi-attention architecture.

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

Tested on Python ≥3.9 and PyTorch ≥2.0.

---

## 🚀 Quick Start

Edit configuration directly in `run.py` under the `config = {...}` section.

Then simply run:

```bash
python run.py
```

This will train and/or forecast using the built-in configuration.

---

## 🧩 Project Structure

| File/Folder                   | Description                                      |
|-------------------------------|--------------------------------------------------|
| `Forecaster/model.py`         | Model definition for QuadraFormer               |
| `Forecaster/data_loader.py`   | Data loading pipeline                           |
| `Forecaster/data_process.py`  | Preprocessing script to generate windowed data  |
| `Forecaster/metrics_evaluation.py` | Metric computation (MSE, MAE, etc.)       |
| `Forecaster/layers/`          | Attention layers and expert modules             |
| `Forecaster/util/`            | Auxiliary functions and time features           |
| `Forecaster/saved_models/`    | Pretrained model checkpoints                    |
| `processed/`                  | Preprocessed data directory                     |
| `run.py`                      | Main training and forecasting script            |
| `README.md`                   | This file                                       |
| `requirements.txt`            | Python dependencies                             |

---

## ⚙️ Example Config (`run.py`)

```python
config = {
    "mode": "train",              # Options: "train", "test", "forecast"
    "interval": 1,                   # Lookback in hours
    "data_set": "SDSS",              # Dataset name: SDSS, tiramisu, alibaba
    "data_type": "hyper",            # Target type: hyper, sql, or resource
    "model_type": "QuadraFormer",    # Model name
    "window_size": 16,
    "prediction_length": 16,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "early_stopping": 1e-3,
    "epochs": 20,
}
```

---

## 📊 Datasets

This project supports the following public workloads:

- **[SDSS Query Log](https://skyserver.sdss.org/log/en/traffic/)**
- **[BusTracker Workload](https://github.com/linmagit/QueryBot5000)**
- **[Alibaba Cluster Data](https://github.com/alibaba/clusterdata)**

All datasets should be preprocessed using `Forecaster/data_process.py`.

---

## 📚 Baseline References

This repo compares against multiple forecasting methods. For reproducibility, we follow their official implementations:

- **PathFormer** ([GitHub](https://github.com/decisionintelligence/pathformer))
- **Sibyl**: Huang et al. *Sibyl: Forecasting Time-Evolving Query Workloads*, ACM SIGMOD 2024.
- **DBAugur** ([GitHub](https://github.com/gaoyuanning/DBAugur))
- **QueryBot5000** ([GitHub](https://github.com/linmagit/QueryBot5000))
- **Vanilla Transformer**: Vaswani et al., NeurIPS 2017

> Only QuadraFormer is included in this repository. For baseline results, please refer to their original codebases.


---

## 📄 License

This repository is released under the MIT License.

Copyright (c) 2025 Anonymous Authors
