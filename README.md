KDMOS:Knowledge Distillation for Motion Segmentation




<p align="center">
<img src="./assets/overview.jpg" width="90%">
</p>
<b><p align="center" style="margin-top: -20px;">
KDMOS
</b></p>



## 📖How to use
### 📦pretrained model
Our pretrained model (best in validation, with the IoU of **_76.12%_**) can be downloaded from [Google Drive](https://drive.google.com/file/d/1KGPwMr9v9GWdIB0zEGAJ8Wi0k3dvXbZt/view?usp=sharing).
### 📚Dataset 
Download SemanticKITTI dataset from [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download) (including **Velodyne point clouds**, **calibration data** and **label data**).Extract everything into the same folder. 
Data file structure should look like this:

```
path_to_KITTI/
├──sequences
    ├── 00/   
    │   ├── calib.txt   # Calibration file.     
    │   ├── poses.txt   # Odometry poses.
    │   ├── velodyne/	# Unzip from KITTI Odometry Benchmark Velodyne point clouds.
    |   |	├── 000000.bin
    |   |	├── 000001.bin
    |   |	└── ...
    │   └── labels/ 	# Unzip from SemanticKITTI label data.
    |       ├── 000000.label
    |       ├── 000001.label
    |       └── ...
    ├── ...
    └── 21/
    └── ...
```
We obtained the residual image according to the [MOTIONBEV](https://github.com/xieKKKi/MotionBEV)

Specify paths in [data_preparing_polar_sequential.yaml](utils/generate_residual/config/data_preparing_polar_sequential.yaml)
```shell
scan_folder: 'your_path/path_to_KITTI/'
residual_image_folder: 'your_path/mos/residual-polar-sequential-480-360/'
```
run
```shell
cd utils/generate_residual/utils
python auto_gen_polar_sequential_residual_images.py
```
Then we obtain motion features with N channels for all scans.

To speed up training, we pre-generated the teacher logits.The data can be downloaded from [Google Drive](https://drive.google.com/file/d/1yht7csVt9C96g38Zqgu2FsHXAzTInoR1/view?usp=drive_link),Or you can manually generate it according to the code in generate_teacher_logits.py and the environment of [MambaMOS](https://github.com/Terminal-K/MambaMOS).






### 💾Environment
This code is tested on Ubuntu 18.04 with Python 3.9, CUDA 11.6 and Pytorch 1.13.0.

Install the following dependencies:
* numpy==1.21.6
* [pytorch](https://pytorch.org/get-started/previous-versions/)==1.13.0+cu116
* tqdm==4.65.0
* pyyaml==6.0
* strictyaml==1.7.3
* icecream==2.1.3
* scipy==1.7.3
* [numba](https://github.com/numba/numba)==0.56.4
* [torch-scatter](https://github.com/rusty1s/pytorch_scatter)==2.1.1+pt113cu116
* [dropblock](https://github.com/miguelvr/dropblock)==0.3.0

### 📈Training
### Train
You may want to modify these params in [KDMOS-semantickitti.yaml](config/KDMOS-semantickitti.yaml)
```shell
data_path: "your_path/path_to_KITTI"
residual_path: "your_path/mos/residual-polar-sequential-480-360/" #"/media/ubuntu/4T/KITTI/mos/residual-polar-sequential-480-360"
teacher_logits_path: "your_path/path_to_teacher_logits"
model_load_path: ""  # none for training from scratch
batch_size: 8
eval_every_n_steps: 1048  #1411 #1048
drop_few_static_frames: True  # drop_few_static_frames for training, speed up training while slightly reduce the accuracy
```
Run
```shell
python train_SemanticKITTI.py
```

### 📝Infer
pretrained models:

[KDMOS-val-79.4.pt](pretrain/KDMOS-val-79.4.pt)  
[KDMOS-road-test-78.8.pt](pretrain/KDMOS-road-test-78.8.pt)


Specify params in [KDMOS-semantickitti.yaml](config/KDMOS-semantickitti.yaml)
```shell
data_path: "your_path/path_to_KITTI"
residual_path: "your_path/mos/residual-polar-sequential-480-360/" #"/media/ubuntu/4T/KITTI/mos/residual-polar-sequential-480-360"
model_load_path: "pretain/MotionBEV-kitti-val-76.54.pt"
```

Run
```shell
python infer_SemanticKITTI.py
```
the predictions will be saved in folder `prediction_save_dir`


### Evaluation
Follow [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api).

Or run:
```shell
python utils/evaluate_mos.py -d your_path/path_to_KITTI -p your_path/path_to_predictions -s valid
```

### Visualization
Install [open3d](https://github.com/isl-org/Open3D) for visualization.
```shell
python utils/visualize_mos.py -d your_path/path_to_KITTI -p your_path/path_to_predictions -s 08
```
## Acknowledgment
We thank for the opensource codebases, [MOTIONBEV](https://github.com/xieKKKi/MotionBEV)
, [LiDAR-MOS](https://github.com/PRBonn/LiDAR-MOS)
, [MotionSeg3D](https://github.com/haomo-ai/MotionSeg3D)
, [MF-MOS](https://github.com/SCNU-RISLAB/MF-MOS)