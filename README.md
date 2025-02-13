# DLFormer
The repo is the official implementation for the paper: [DLFormer: Enhancing Explainability in Time Series Forecasting using Distributed Lag Machanism](https://arxiv.org/abs/2408.16896)
# Usage
**Installation**

Step1: Create a conda environment and activate it
```
conda create -n DLFormer python==3.8 --y
conda activate fontdiffuser
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

**Training & Test**
```
python main.py --data SML --pred 96 --seq 48 --batch_size 32 --log_interval 1 --plot_interval 30
```

**Results**

You can check the plots for validation and test in the results subfolder!
