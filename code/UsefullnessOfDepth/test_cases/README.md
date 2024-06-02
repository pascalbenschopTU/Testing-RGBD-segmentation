
# FgBG

DFormer
```
python .\test_cases\test_fgbg_aux.py -d .\datasets\test_suite_fgbg_large_spacing_appearance\ -chdir checkpoints_fgbg_spacing -c .\configs\SynthDet\SynthDet_foreground_background_test_pretrained_aux.py -mti 200
```

TokenFusion:
```
python .\test_cases\test_fgbg.py -d .\datasets\test_suite_fgbg_large_spacing_appearance\ -chdir checkpoints_fgbg_spacing_TF -c .\configs\SynthDet\SynthDet_foreground_background_test_pretrained_aux_TF.py -mti 200 -m TokenFusion
```

# Spatial

DFormer
```
python .\test_cases\test_spatial.py -d .\datasets\test_suite_spatial_realistic_no_walls\ -chdir checkpoints_spatial_groceries -dc groceries -c .\configs\SynthDet\SynthDet_spatial_test_pretrained.py -mti 200
```

TokenFusion
```
python .\test_cases\test_spatial.py -d .\datasets\test_suite_spatial_realistic_no_walls\ -chdir checkpoints_spatial_groceries_TF -dc groceries -c .\configs\SynthDet\SynthDet_spatial_test_pretrained_TF.py -mti 200 -mn rgbd -m TokenFusion
```

# Robustness

TokenFusion
```
python .\test_cases\test_robustness_TF.py -d .\datasets\test_suite_robustness\SynthDet_robustness_groceries\ -chdir checkpoints_robustness_groceries_non_rot_TF -dc groceries -c .\configs\SynthDet\SynthDet_robustness_test_pretrained_TF.py -m TokenFusion
```


### Random
```
python .\test_cases\test_robustness.py -d .\datasets\test_suite_robustness\SynthDet_rgb -chdir checkpoints_robustness_pretrained
-dc random -he 0 -e 30 -c .\configs\SynthDet\SynthDet_robustness_test_pretrained.py

python .\test_cases\test_robustness.py -d .\datasets\test_suite_robustness\SynthDet_red_green_blue_gems\ -chdir checkpoints_robustness -dc random -he 3 -e 30 -c .\configs\SynthDet\SynthDet_robustness_test.py


python .\test_cases\test_robustness.py -d .\datasets\test_suite_robustness\SynthDet_robustness_groceries\ -chdir checkpoints_robustness_groceries -dc random -c .\configs\SynthDet\SynthDet_robustness_test.py

python .\test_cases\test_robustness_TF.py -d .\datasets\test_suite_robustness\SynthDet_robustness_groceries\ -chdir checkpoints_robustness_groceries_non_rot_TF -dc groceries -c .\configs\SynthDet\SynthDet_robustness_test_pretrained_TF.py -m TokenFusion -l G:\mscPascalBenschop\code\UsefullnessOfDepth\checkpoints_robustness_groceries_non_rot_TF\log_20240602_111554.txt
```