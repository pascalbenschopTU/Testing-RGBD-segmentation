# MscPascalBenschop: Why is Depth useful for Computer Vision?

Repository for the master thesis project of Pascal Benschop. 



## Experiments

### Environment
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
```
#### Instal torch

For GPU: (Download CUDA 11.8)
```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/test/cu118
```
For CPU:
```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/test/cpu
```

#### Install mmcv

For GPU:
```
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
```
For CPU:
```
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html
```

Install Ray tune for hyperparameter optimization:
```
pip install "ray[tune]"
```


### Pretrained weights
For the SegFormer mit weights, use this [google drive link](https://drive.google.com/drive/folders/10XgSW8f7ghRs9fJ0dE-EV8G2E_guVsT5?usp=sharing) from the [CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers](https://github.com/huaaaliu/RGBX_Semantic_Segmentation) repository.

For the DFormer weights, use this [google drive link](https://drive.google.com/drive/folders/1YuW7qUtnguUFkhC-sfqGySrerjK0rZJX?usp=sharing) from the [DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation](https://github.com/VCIP-RGBD/DFormer) repository.

Then store all weights under a folder `UsefulnessOfDepth/checkpoints/pretrained`. See for example the [model configurations](code/UsefullnessOfDepth/configs/model_configurations.json) file.

### Model setup
Place model files in a folder under `UsefulnessOfDepth`.
Then add the model to [model_wrapper.py](code/UsefullnessOfDepth/utils/model_wrapper.py) in the `set_model` function. Also adjust the `forward` and `get_loss` function if the model has a specialized forward function / loss calculation.

### Dataset setup

The test suites and the background dataset can be downloaded as a zip from this [google drive link](https://drive.google.com/file/d/1PxgtCENRhFFNqKPtmzWrFyxlclGmmGw9/view?usp=drive_link). Place the folders under `UsefulnessOfDepth/datasets`.

The NYUDepthV2 (and SUNRGBD) datasets can be downloaded from this [google drive link](https://drive.google.com/drive/folders/1RIa9t7Wi4krq0YcgjR3EWBxWWJedrYUl) from the [DFormer](https://github.com/VCIP-RGBD/DFormer) repository. Place these folders under `UsefulnessOfDepth/datasets`.

### Training

Then follow the instructions in the [README](code/UsefullnessOfDepth/test_cases/README.md) for testing the model on its robustness against variations in RGB, spatial layouts in Depth and changes in backgrounds with similar foregrounds.

## SynthDet

### Setup

Clone SynthDet from unity: https://github.com/Unity-Technologies/SynthDet

Open a Unity project with the folder. 
Select the SynthDet scene from the `scenes` folder in Assets if it is not open.
Click on the Main Camera in the Hierarchy menu. Under `Perception Camera (Script)` click on the `+` symbol under `Camera Labelers` and select the `Depth Labeler`.

Then copy the scripts from `UsefullnessOfDepth/SynthDet_tools/Randomizers` to The Unity project `path/to/SynthDet/Assets/Scripts/Randomizers`.



## Sources

- **[SynthDet](https://github.com/Unity-Technologies/SynthDet)**: A project by Unity Technologies for synthetic data generation and object detection.
- **[DFormer](https://github.com/VCIP-RGBD/DFormer)**: A repository for DFormer, a pretraining framework for a RGB-D transformer-based model.
- **[TokenFusion](https://github.com/yikaiw/TokenFusion)**: TokenFusion focuses on efficient multi-modal token fusion for vision tasks.
- **[CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation)**: CMX is a project dedicated to RGB-X semantic segmentation.
- **[Gemini](https://github.com/JiaDingCN/GeminiFusion/tree/main)**: The GeminiFusion project is aimed at multi-sensor fusion using the Gemini framework.
- **[HiDANet](https://github.com/Zongwei97/HIDANet/tree/main)**: HiDANet is a repository for High-Resolution Depth-Aware Network for salient object detection.



