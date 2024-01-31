
## Analyzing results

Template:

```
python compare_results.py --model_a_path ... \
--model_b_path ... \
--dir_dataset ... \
--channels_a . \
--channels_b . \
--config ..
```

```
python compare_results.py --model_a_path checkpoints\SynthDet_default_Segmodelc3_20240130-165429\checkpoint_epoch_90_miou_91.51.pth --model_b_path checkpoints\SynthDet_default_Segmodelc4_20240130-112340\checkpoint_epoch_90_miou_92.54.pth --dir_dataset ..\..\DFormer\datasets\SynthDet_default\ --channels_a 3 --channels_b 4 --config config.SynthDet_default_Segmodel

```
