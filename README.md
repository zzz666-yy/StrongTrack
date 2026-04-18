# StrongTrack


## 1.  Installation

**Environmental requirements：**
- CUDA 11.7
- Python 3.8

```bash
# Install the basic dependencies
pip install -r requirements.txt

python setup.py develop

# Install the necessary third-party libraries
pip install pycocotools cython_bbox faiss-gpu

## 2.  Data Preparation
datasets
├── mot
│   ├── train
│   └── test
└── MOT20
    ├── train
    └── test
After organizing the data, run the following script to convert the dataset into the COCO format.
python tools/convert_mot17_to_coco.py
python tools/convert_mot20_to_coco.py
## 3.  Tracking
Evaluation on MOT17 half val
python tools/track.py -f exps/example/mot/yolox_x_ablation.py \
    -c pretrained/strongtrack_ablation.pth.tar \
    -b 1 -d 1 --fp16 --fuse \
    --cmc-method sparseOptFlow \
    --with_reid \
    --fast_reid_config fast_reid/configs/MOT17/sbs_S50.yml \
    --fast_reid_weights pretrained/fastreid.pth
You need to use TrackEval to evaluate the code and process the generated experimental results, so as to obtain the various indicator data mentioned in the text.
Test on MOT17
python tools/track.py -f exps/example/mot/yolox_x_mix_det.py \
    -c pretrained/strongtrack_x_mot17.pth.tar \
    -b 1 -d 1 --fp16 --fuse \
    --cmc-method sparseOptFlow \
    --with_reid \
    --fast_reid_config fast_reid/configs/MOT17/sbs_S50.yml \
    --fast_reid_weights pretrained/fastreid.pth
python tools/interpolation.py
Test on MOT20
python tools/track.py -f exps/example/mot/yolox_x_mix_mot20_ch.py \
    -c pretrained/strongtrack_x_mot20.pth.tar \
    -b 1 -d 1 --fp16 --fuse \
    --match_thresh 0.7 --mot20 \
    --cmc-method sparseOptFlow \
    --with_reid \
    --fast_reid_config fast_reid/configs/MOT17/sbs_S50.yml \
    --fast_reid_weights pretrained/fastreid.pth
python tools/interpolation.py
For the MOT test set, you need to upload the tracking results to the MOTChallenge official website.








