# StrongTrack

## 1. Installation

It is recommended to use Anaconda for environment management on the host machine.

**Environmental requirements:**
- CUDA 11.7
- Python 3.8

### Install the basic dependencies
pip install -r requirements.txt
    
### Build the project
python setup.py develop
    
### Install the necessary third-party libraries
pip install pycocotools cython_bbox faiss-gpu

---

## 2. Data Preparation

Please organize your tracking datasets into the following directory structure:

    datasets
    ├── mot
    │   ├── train
    │   └── test
    └── MOT20
        ├── train
        └── test

After organizing the data, run the following script to convert the dataset into the COCO format:

    python tools/convert_mot17_to_coco.py
    python tools/convert_mot20_to_coco.py

---

---

## 3. Model Zoo

Please download the pretrained models and place them in the `pretrained` folder:

    <StrongTrack_dir>/pretrained

The expected directory structure is:

    pretrained
    ├── strongtrack_ablation.pth.tar
    ├── strongtrack_x_mot17.pth.tar
    ├── strongtrack_x_mot20.pth.tar
    └── fastreid.pth

| Model | Dataset / Usage | File Name | Download |
|---|---|---|---|
| StrongTrack Ablation | MOT17 half-val evaluation | `strongtrack_ablation.pth.tar` | [Download]( https://pan.baidu.com/s/1hDqT1_oVps1bVmAr-HZ8OQ?pwd=yxpy) |
| StrongTrack MOT17 | MOT17 test set | `strongtrack_x_mot17.pth.tar` | [Download]( https://pan.baidu.com/s/1hDqT1_oVps1bVmAr-HZ8OQ?pwd=yxpy) |
| StrongTrack MOT20 | MOT20 test set | `strongtrack_x_mot20.pth.tar` | [Download]( https://pan.baidu.com/s/1hDqT1_oVps1bVmAr-HZ8OQ?pwd=yxpy) |
| FastReID | ReID feature extraction | `fastreid.pth` | [Download]( https://pan.baidu.com/s/1hDqT1_oVps1bVmAr-HZ8OQ?pwd=yxpy) |

> **Note:** Please make sure the downloaded model file names are consistent with the paths used in the tracking commands, such as `pretrained/strongtrack_x_mot17.pth.tar` and `pretrained/fastreid.pth`.

---

## 4. Tracking

### 4.1 Evaluation on MOT17 half val

    python tools/track.py -f exps/example/mot/yolox_x_ablation.py \
        -c pretrained/strongtrack_ablation.pth.tar \
        -b 1 -d 1 --fp16 --fuse \
        --cmc-method sparseOptFlow \
        --with_reid \
        --fast_reid_config fast_reid/configs/MOT17/sbs_S50.yml \
        --fast_reid_weights pretrained/fastreid.pth

> **Note:** You need to use TrackEval to evaluate the code and process the generated experimental results, so as to obtain the various indicators mentioned in the text.

### 4.2 Test on MOT17

    # Run tracking
    python tools/track.py -f exps/example/mot/yolox_x_mix_det.py \
        -c pretrained/strongtrack_x_mot17.pth.tar \
        -b 1 -d 1 --fp16 --fuse \
        --cmc-method sparseOptFlow \
        --with_reid \
        --fast_reid_config fast_reid/configs/MOT17/sbs_S50.yml \
        --fast_reid_weights pretrained/fastreid.pth
    
    # Run interpolation
    python tools/interpolation.py

### 4.3 Test on MOT20

    # Run tracking
    python tools/track.py -f exps/example/mot/yolox_x_mix_mot20_ch.py \
        -c pretrained/strongtrack_x_mot20.pth.tar \
        -b 1 -d 1 --fp16 --fuse \
        --match_thresh 0.7 --mot20 \
        --cmc-method sparseOptFlow \
        --with_reid \
        --fast_reid_config fast_reid/configs/MOT17/sbs_S50.yml \
        --fast_reid_weights pretrained/fastreid.pth
    
    # Run interpolation
    python tools/interpolation.py

> **Note:** For the MOT test set, you need to upload the tracking results to the MOTChallenge official website.








