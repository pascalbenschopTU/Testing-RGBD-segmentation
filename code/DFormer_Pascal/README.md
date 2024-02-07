# Plotting the predictions:
```
cd ..\DFormer_Pascal

python .\plot_predictions.py --dir_rgb ..\DFormer\checkpoints\SynthDet2_DFormer-Tiny_20240121-162131\ --dir_rgbd ..\DFormer\checkpoints\SynthDet2_DFormer-Tiny_20240121-130329\ --dir_dataset ..\DFormer\datasets\SynthDet2\

python .\plot_predictions.py --dir_rgb ..\DFormer\checkpoints\SynthDet3_DFormer-Tiny_20240123-114720\ --dir_rgbd ..\DFormer\checkpoints\SynthDet3_DFormer-Tiny_20240124-081538\ --dir_dataset ..\DFormer\datasets\SynthDet3\

python .\plot_predictions.py --dir_rgb ..\DFormer\checkpoints\SynthDet_default_DFormer-Tiny_20240124-122317\ --dir_rgbd ..\DFormer\checkpoints\SynthDet_default_DFormer-Tiny_20240124-143945\ --dir_dataset ..\DFormer\datasets\SynthDet_default\

python .\plot_predictions.py --dir_rgb ..\DFormer\checkpoints\SynthDet_large_DFormer-Tiny_20240127-121104\ --dir_rgbd ..\src\SynthDet_Segmentation\checkpoints\SynthDet_large_Segmodel_20240128-145956\ --dir_dataset ..\DFormer\datasets\SynthDet_large\

python .\plot_predictions.py --dir_rgb ..\src\SynthDet_Segmentation\checkpoints\SynthDet3_Segmodelc3_20240129-122809\ --dir_rgbd ..\src\SynthDet_Segmentation\checkpoints\SynthDet3_Segmodelc4_20240129-113204\ --dir_dataset ..\DFormer\datasets\SynthDet3\
```

# Plotting the accuracy:

```
python plot_accuracy.py --dir_rgb ..\DFormer\checkpoints\SynthDet2_DFormer-Tiny_20240121-162131\ --dir_rgbd ..\DFormer\checkpoints\SynthDet2_DFormer-Tiny_20240121-130329\
```


# Creating results (inside DFormer)

```
python .\utils\create_predictions.py --config local_configs.SynthDet.SynthDet_gems_light_color_DFormer_Tiny --model_path .\checkpoints\SynthDet_gems_light_color_DFormer-Tiny_20240205-125951\epoch-100.pth_miou_7.24 --output_path .\checkpoints\SynthDet_gems_light_color_DFormer-Tiny_20240205-125951\predictions

```