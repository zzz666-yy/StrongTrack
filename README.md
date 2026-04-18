1. Installing on the host machine
    Use the Anaconda (CUDA 11.7   python 3.8)
    pip install -r requirements.txt
    python setup.py develop
    Install pycocotools cython_bbox  faiss-gpu

2. Data Preparation
datasets in the following structure:

datasets
   |——————mot
   |        └——————train
   |        └——————test
   └——————MOT20
            └——————train
            └——————test
     
   
Then, you need to turn the datasets to COCO format and mix different training data:
python tools/convert_mot17_to_coco.py
python tools/convert_mot20_to_coco.py

3. Tracking
 3.1 Evaluation on MOT17 half val
 python tools/track.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/strongtrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse --cmc-method sparseOptFlow --with_reid --fast_reid_config D:\MOT\Strongtrack\fast_reid\configs\MOT17\sbs_S50.yml --fast_reid_weights pretrained/fastreid.pth
 You need to evaluate the experimental results using the TrackEVAL assessment code, and this will enable you to obtain the data mentioned in the text.
 3.2 Test on MOT17
 python tools/track.py -f exps/example/mot/yolox_x_mix_det.py -c pretrained/strongtrack_x_mot17.pth.tar -b 1 -d 1 --fp16 --fuse --cmc-method sparseOptFlow --with_reid --fast_reid_config D:\MOT\Strongtrack\fast_reid\configs\MOT17\sbs_S50.yml --fast_reid_weights pretrained/fastreid.pth
 python tools/interpolation.py
 3.3 Test on MOT20
 python tools/track.py -f exps/example/mot/yolox_x_mix_mot20_ch.py -c pretrained/strongtrack_x_mot20.pth.tar -b 1 -d 1 --fp16 --fuse --match_thresh 0.7 --mot20 --cmc-method sparseOptFlow --with_reid --fast_reid_config D:\MOT\Strongtrack\fast_reid\configs\MOT17\sbs_S50.yml --fast_reid_weights pretrained/fastreid.pth
 python tools/interpolation.py
 For the MOT test set, you need to upload the tracking results to the MOTChallenge official website.
