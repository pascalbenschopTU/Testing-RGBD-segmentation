To use adapt_dataset_and_test.py copy the RGB folder to another folder and run the following command:

```
python .\utils\adapt_dataset_and_test.py ^
--config=configs.SUNRGBD.SUNRGBD_DFormer_Tiny ^
--model_weights .\checkpoints\SUNRGBD_DFormer-Large\run_20240427-205208\epoch_30_miou_25.014.pth ^
--model {model_type, choose from [DFormer-Large, DeepLab, CMX]} ^
--bin_size {dataset_length or smaller} ^
--property_name {property_name}
```

An example is the following:
```
python .\utils\adapt_dataset_and_test.py ^
--config=configs.SUNRGBD.SUNRGBD_DFormer_Tiny ^
--model_weights .\checkpoints\SUNRGBD_DFormer-Large\run_20240427-205208\epoch_30_miou_25.014.pth ^
--model DFormer-Large ^
--bin_size 5050 ^
--property_name saturation
```


For gradcam install gradcam locally from the folder pytorch-grad-cam (add this folder to project)

```
python .\utils\dformer_gradcam.py ^
-cfg .\configs\SynthDet\SynthDet_cars_foreground_background_DFormer.py  ^
-mp .\checkpoints\SynthDet_cars_foreground_background_DFormer-Tiny\run_20240506-125632_depth_augmented\epoch_60 _miou_13.031.pth ^
-d .\datasets\SynthDet_carse_foreground_background_diff_white\ ^
-a ^
-t 0
```
