# MscPascalBenschop: Why is Depth useful for Computer Vision?

Repository for the master thesis project of Pascal Benschop. 

## SynthDet

### Setup

clone SynthDet from unity: https://github.com/Unity-Technologies/SynthDet

Run simulations via Unity, enable depth labeler.

Activate conda environment:

Step 1
```
conda env create -f environment.yaml
```
If an error occurs with something like: `An error occurred while installing package 'defaults::xz-5.4.5-h8cc25b3_0'.
Rolling back transaction: done`

Then do `conda env remove --name python_environment` and `conda env create -f environment.yaml`

Step 2
```
conda activate python_environment

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html

pip install datasetinsights
```

For Grad-CAM

Navigate to pytorch-grad-cam, and once inside:
```
pip install .
```

### Training

Navigate to DFormer repository
```
python .\utils\train.py --config=path/to/config.py
```


## Sources

[SynthDet](https://github.com/Unity-Technologies/SynthDet)
[2.5 Malleable Convolution](https://github.com/charlesCXK/RGBD_Semantic_Segmentation_PyTorch)


Apparently bincount is very slow with a lot of the same class https://discuss.pytorch.org/t/torch-bincount-1000x-slower-on-cuda/42654/2 


