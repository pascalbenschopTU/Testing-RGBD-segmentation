
## Creating results

```
python .\create_predictions.py --channels 3 --config config.SynthDet_gems_light_color_Segmodel --model_path .\checkpoints\SynthDet_gems_light_color_Segmodelc3_20240205-111412\checkpoint_epoch_100_miou_12.99.pth --output_path .\checkpoints\SynthDet_gems_light_color_Segmodelc3_20240205-111412\predictions

python .\create_predictions.py --channels 4 --config config.SynthDet_gems_light_color_Segmodel --model_path .\checkpoints\SynthDet_gems_light_color_Segmodelc4_20240205-120328\checkpoint_epoch_100_miou_14.76.pth --output_path .\checkpoints\SynthDet_gems_light_color_Segmodelc4_20240205-120328\predictions
```


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


python .\compare_results.py --model_a_path .\checkpoints\SynthDet_shadows_Segmodelc3_20240131-135239\checkpoint_epoch_100_miou_87.66.pth --model_b_path .\checkpoints\SynthDet_shadows_Segmodelc4_20240131-145942\checkpoint_epoch_100_miou_90.75.pth --dir_dataset ..\..\DFormer\datasets\SynthDet_shadows\ --channels_a 3 --channels_b 4 --config config.SynthDet_shadows_Segmodel

```

### Comparing SynthDet gems
```


python .\compare_results.py --model_a_path .\checkpoints\SynthDet_gems_light_color_Segmodelc3_20240205-111412\ --model_b_path .\checkpoints\SynthDet_gems_light_color_Segmodelc4_20240205-120328\ --dir_dataset ..\..\DFormer\datasets\SynthDet_gems_light_color\ --channels_a 3 --channels_b 4 --config config.SynthDet_gems_light_color_Segmodel

python .\compare_results.py --model_a_path ..\..\DFormer\checkpoints\SynthDet_gems_light_color_DFormer-Tiny_20240205-125951\ --model_b_path .\checkpoints\SynthDet_gems_light_color_Segmodelc4_20240205-120328\ --dir_dataset ..\..\DFormer\datasets\SynthDet_gems_light_color\ --channels_a 4 --channels_b 4 --config config.SynthDet_gems_light_color_Segmodel
```
