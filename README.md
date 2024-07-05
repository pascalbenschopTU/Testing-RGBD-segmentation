# MscPascalBenschop: Why is Depth useful for Computer Vision?

Repository for the master thesis project of Pascal Benschop. 

## SynthDet

### Setup

Clone SynthDet from unity: https://github.com/Unity-Technologies/SynthDet

Open a Unity project with the folder. 
Select the SynthDet scene from the `scenes` folder in Assets if it is not open.
Click on the Main Camera in the Hierarchy menu. Under `Perception Camera (Script)` click on the `+` symbol under `Camera Labelers` and select the `Depth Labeler`.

Then import the scripts from <TODO>


## Experiments
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


### Training

Place model files in a folder under `UsefulnessOfDepth`.
Then add the model to `UsefulnessOfDepth\utils\model_wrapper.py` in the `set_model` function. Also adjust the forward and get_loss function if the model has a specialized forward function / loss calculation.

Then follow the instructions in the [README](code\UsefullnessOfDepth\test_cases\README.md) for testing the model on its robustness against variations in RGB, spatial layouts in Depth and changes in backgrounds with similar foregrounds.


## Sources

[SynthDet](https://github.com/Unity-Technologies/SynthDet)
[DFormer](https://github.com/VCIP-RGBD/DFormer)
[TokenFusion](https://github.com/yikaiw/TokenFusion)
[CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation)
[Gemini](https://github.com/JiaDingCN/GeminiFusion/tree/main)
[HiDANet](https://github.com/Zongwei97/HIDANet/tree/main)


