# DLFormer - Pytorch Implementation
The repo is the official implementation for the paper: [Distributed Lag Transformer based on Time-Variable-Aware Learning for Explainable Multivariate Time Series Forecasting]()

# Architecture
**Overall Structure**
![Image](https://github.com/user-attachments/assets/2a8c5705-d833-4cb2-871a-9749450af9e2)
**DL Embedding**
![Image](https://github.com/user-attachments/assets/813a3b69-cac5-42c1-bf7f-1b062f71efce)

# Usage
**Installation**

Step1: Create a conda environment and activate it
```
conda create -n DLFormer python==3.8 --y
conda activate DLFormer
```
Step2: Install related version Pytorch following [here](https://pytorch.org/get-started/previous-versions/).
```
# Suggested
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
Step3: Install the required packages.
```
pip install -r requirements.txt
```

**Train & Test**
```
python main.py --data SML --pred 96 --seq 48 --batch_size 32 --log_interval 1 --plot_interval 30
```

# Explainability

**Global Explainability**
![Image](https://github.com/user-attachments/assets/85816788-4de0-4de6-85c2-c3dd2526b51a)
**Local Explainability**
![Image](https://github.com/user-attachments/assets/b7365dbb-21a3-4fb0-b615-8caaaf3fbaff)

# Notice

- You can check the plots for validation and test in the results subfolder!
- Due to the DLFormer’s Temporal-Variate-Aware Learning strategy, the optimization process takes a long time. Please maintain approximately 500 epochs. :smile:
