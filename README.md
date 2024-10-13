# This is a clean implementation of DDPM
<img src='https://github.com/MyRepositories-hub/clean_ddpm/blob/main/results/enjoy.gif' width='100%'/>

## Installation
Create Anaconda environment
```bash
conda create -n ddpm_py311 python=3.11 --yes
conda activate ddpm_py311
```
Choose the CUDA version on the official PyTorch website: [https://pytorch.org/](https://pytorch.org/)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install the remaining requirements
```bash
pip install -r requirements.txt
```

## Train DDPM on MNIST
```bash
python main.py
```

## Enjoy
```bash
python inference.py
```
## More results
<img src='https://github.com/MyRepositories-hub/clean_ddpm/blob/main/results/enjoy_0.gif' width='50%'/><img src='https://github.com/MyRepositories-hub/clean_ddpm/blob/main/results/enjoy_1.gif' width='50%'/>
<img src='https://github.com/MyRepositories-hub/clean_ddpm/blob/main/results/enjoy_2.gif' width='50%'/><img src='https://github.com/MyRepositories-hub/clean_ddpm/blob/main/results/enjoy_3.gif' width='50%'/>
<img src='https://github.com/MyRepositories-hub/clean_ddpm/blob/main/results/enjoy_4.gif' width='50%'/><img src='https://github.com/MyRepositories-hub/clean_ddpm/blob/main/results/enjoy_5.gif' width='50%'/>
<img src='https://github.com/MyRepositories-hub/clean_ddpm/blob/main/results/enjoy_6.gif' width='50%'/><img src='https://github.com/MyRepositories-hub/clean_ddpm/blob/main/results/enjoy_7.gif' width='50%'/>
<img src='https://github.com/MyRepositories-hub/clean_ddpm/blob/main/results/enjoy_8.gif' width='50%'/><img src='https://github.com/MyRepositories-hub/clean_ddpm/blob/main/results/enjoy_9.gif' width='50%'/>
