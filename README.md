# StrongTrack


## 1. 环境安装 (Installation)


**环境要求：**
- CUDA 11.7
- Python 3.8

```bash
# 安装基础依赖
pip install -r requirements.txt

python setup.py develop

# 安装必要的第三方库
pip install pycocotools cython_bbox faiss-gpu

Data Preparation
datasets
├── mot
│   ├── train
│   └── test
└── MOT20
    ├── train
    └── test
