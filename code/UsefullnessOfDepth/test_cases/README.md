
# FgBG

DFormer
```
python .\test_cases\test_fgbg.py -d .\datasets\test_suite_fgbg_large_spacing_appearance\ -chdir checkpoints_fgbg_spacing -c .\configs\SynthDet\SynthDet_foreground_background_test_pretrained_aux.py -mti 200 -mn rgbd rgbd_aux -m DFormer
```

TokenFusion:
```
python .\test_cases\test_fgbg.py -d .\datasets\test_suite_fgbg_large_spacing_appearance\ -chdir checkpoints_fgbg_spacing_TF -c .\configs\SynthDet\SynthDet_foreground_background_test_pretrained_TF.py -mti 200 -m TokenFusion -mn rgbd
```

SegFormer:
```
python .\test_cases\test_fgbg.py -d .\datasets\test_suite_fgbg_large_spacing_appearance\ -chdir checkpoints_fgbg_spacing_SF -c .\configs\SynthDet\SynthDet_foreground_background_test_pretrained_SF.py -mti 200 -m SegFormer -mn rgb
```

# Spatial

DFormer
```
python .\test_cases\test_spatial.py -d .\datasets\test_suite_spatial_realistic_no_walls\ -chdir checkpoints_spatial_groceries -dc groceries -c .\configs\SynthDet\SynthDet_spatial_test_pretrained.py -mti 200 -mn rgbd -m DFormer
```

TokenFusion
```
python .\test_cases\test_spatial.py -d .\datasets\test_suite_spatial_realistic_no_walls\ -chdir checkpoints_spatial_groceries_TF -dc groceries -c .\configs\SynthDet\SynthDet_spatial_test_pretrained_TF.py -mti 200 -mn rgbd -m TokenFusion
```

SegFormer
```
python .\test_cases\test_spatial.py -d .\datasets\test_suite_spatial_realistic_no_walls\ -chdir checkpoints_spatial_groceries_SF -dc groceries -c .\configs\SynthDet\SynthDet_spatial_test_pretrained_SF.py -mti 200 -mn rgb -m SegFormer
```

CMX
```
python .\test_cases\test_spatial.py -d .\datasets\test_suite_spatial_realistic_no_walls\ -chdir checkpoints_spatial_groceries_CMX -dc groceries -c .\configs\SynthDet\SynthDet_spatial_test_pretrained_CMX.py -mti 200 -mn rgb -m CMX
```


# Robustness

DFormer
```
python .\test_cases\test_robustness.py -d .\datasets\test_suite_robustness\SynthDet_robustness_groceries\ -chdir checkpoints_robustness_groceries -dc groceries -c .\configs\SynthDet\SynthDet_robustness_test_pretrained.py -mn rgbd rgbd_variation -m DFormer
```

TokenFusion
```
python .\test_cases\test_robustness.py -d .\datasets\test_suite_robustness\SynthDet_robustness_groceries\ -chdir checkpoints_robustness_groceries_TF -dc groceries -c .\configs\SynthDet\SynthDet_robustness_test_pretrained_TF.py -mn rgbd rgbd_variation -m TokenFusion
```

SegFormer
```
python .\test_cases\test_robustness.py -d .\datasets\test_suite_robustness\SynthDet_robustness_groceries\ -chdir checkpoints_robustness_groceries_SF -dc groceries -c .\configs\SynthDet\SynthDet_robustness_test_pretrained_SF.py -mn rgb rgb_variation -m SegFormer
```

DeepLab
```
python .\test_cases\test_robustness.py -d .\datasets\test_suite_robustness\SynthDet_robustness_groceries\ -chdir checkpoints_robustness_groceries_DL -dc groceries -c .\configs\SynthDet\SynthDet_robustness_test_pretrained_DL.py -mn rgb rgb_variation -m DeepLab
```

CMX
```
python .\test_cases\test_robustness.py -d .\datasets\test_suite_robustness\SynthDet_robustness_groceries\ -chdir checkpoints_robustness_groceries_CMX -dc groceries -c .\configs\SynthDet\SynthDet_robustness_test_pretrained_CMX.py -mn rgbd rgbd_variation -m CMX

```


### NYUDV2

Foreground background tests on NYUDv2
```
python .\test_cases\test_fgbg_NYUDV2.py -c .\configs\NYUDepthv2\DFormer_Base.py -tc .\configs\NYUDepthv2\DFormer_Base_test.py -mw .\checkpoints\NYUDepthv2_DFormer-Base\run_20240607-111847\epoch_100_miou_46.619.pth -m DFormer -chdir checkpoints_fgbg_NYUDV2 -bgdp .\datasets\background\ -redp .\datasets\NYUDepthv3\

python .\test_cases\test_fgbg_NYUDV2.py -c .\configs\NYUDepthv2\SegFormer.py -tc .\configs\NYUDepthv2\SegFormer_test.py -mw .\checkpoints\NYUDepthv2_MiT-B2\run_20240606-222137\epoch_100_miou_42.286.pth -m SegFormer -chdir checkpoints_fgbg_NYUDV2 -bgdp .\datasets\background\ -redp .\datasets\NYUDepthv3\
```

Spatial tests for depth range (-dr) 0.1, 0.2, 0.33

```
python .\utils\adapt_dataset_and_test.py -op .\datasets\NYUDepthv2\Depth_original\ -dp .\datasets\NYUDepthv2\Depth\ -cfg .\configs\NYUDepthv2\DFormer_Base.py -mw .\checkpoints\NYUDepthv2_DFormer-Base\run_20240607-111847\epoch_100_miou_46.619.pth -m DFormer -bs 1000 -pname depth_level -pmin 0.0 -pmax 0.9 -s empty -dr 0.1

python .\utils\adapt_dataset_and_test.py -op .\datasets\NYUDepthv2\Depth_original\ -dp .\datasets\NYUDepthv2\Depth\ -cfg .\configs\NYUDepthv2\DFormer_Base.py -mw .\checkpoints\NYUDepthv2_DFormer-Base\run_20240607-111847\epoch_100_miou_46.619.pth -m DFormer -bs 1000 -pname depth_level -pmin 0.0 -pmax 0.66666 -pvr 3 -s empty -dr 0.333
```

Test cases for adjusting background manually:

```
python .\utils\adjust_background_dataset.py -bgdp .\datasets\background\ -redp .\datasets\NYUDepthv3\ -c .\configs\NYUDepthv2\DFormer_Base.py -cl 31

python .\utils\adjust_background_dataset.py -bgdp .\datasets\background\ -redp .\datasets\NYUDepthv4\ -c .\configs\NYUDepthv2\DFormer_Base.py -cl 31 -o
```

And Predicting examples

```
python .\utils\create_predictions.py --model DFormer --config .\configs\NYUDepthv2\DFormer_Base_test.py --model_weights .\checkpoints\NYUDepthv2_DFormer-Base\run_20240607-111847\epoch_100_miou_46.619.pth --ignore_background --prediction_folder predictions -sc 14 --dataset .\datasets\NYUDepthv4\
```