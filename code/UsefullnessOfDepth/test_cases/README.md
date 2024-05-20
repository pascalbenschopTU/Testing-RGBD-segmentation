Run with: 

```
python .\test_cases\test_foreground_background_separation.py -d .\datasets\test_suite_foreground_background\ -chdir checkpoints_fgbg_pretrained 
-dc random -he 0 -e 30 -c .\configs\SynthDet\SynthDet_foreground_background_test_pretrained.py

python .\test_cases\test_robustness.py -d .\datasets\test_suite_robustness\SynthDet_rgb -chdir checkpoints_robustness_pretrained
-dc random -he 0 -e 30 -c .\configs\SynthDet\SynthDet_robustness_test_pretrained.py

```