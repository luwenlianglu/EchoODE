# Implementation of Echo-ODE
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v1.13+-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/python-v3.6+-blue.svg?logo=python&style=for-the-badge" /></a>

## Quick start

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch 2.0.1 or later](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download the data and run training:
```bash
python main.py --amp
```
## Citation
Original paper by Wenliang Lu:

[Echo-ODE: A Dynamics Modeling Network with Neural ODE for Temporally Consistent Segmentation of Video Echocardiograms](https://github.com/luwenlianglu/EchoODE)
## network architecture
![network architecture](https://github.com/luwenlianglu/EchoODE/blob/master/overview.png)
