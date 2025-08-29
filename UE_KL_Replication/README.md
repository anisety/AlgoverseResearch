# Unsupervised Elicitation KL Divergence Replication

This project replicates the **Unsupervised Elicitation (UE) paper** using **KL divergence** as a metric for model consistency.

## 🔹 Project Overview
- Implement a simple ML model (`SimpleUEModel`) to predict distributions
- Measure alignment using KL divergence (`metrics/kl_divergence.py`)
- Run training loops and record results

## 🚀 Features
- KL divergence computation between model predictions and target distributions
- Modular PyTorch implementation
- Example trial included

## 🛠️ Installation
```bash
git clone <repo_url>
cd UE_KL_Replication
pip install -r requirements.txt
```

## 📂 Project Structure
See folder structure above.

## ▶️ Running Trial 1
```bash
python main.py
```

## Sample Trial 1 Results
KL divergence per epoch: [0.6523, 0.5214, 0.4789, 0.4501, 0.4312]

Notes: Model steadily aligns with target distributions.

## 🔮 Next Steps
- Integrate real UE datasets
- Experiment with larger architectures
- Evaluate consistency across multiple trials
- Use KL divergence as an unsupervised skill elicitation metric
