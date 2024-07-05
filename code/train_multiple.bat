
cd UsefullnessOfDepth

@REM call python .\test_cases\test_robustness.py -d .\datasets\test_suite_robustness\SynthDet_robustness_groceries\ -chdir checkpoints_robustness_groceries -dc groceries -c .\configs\SynthDet\SynthDet_robustness_test_pretrained.py -mn rgbd rgbd_variation depth -m DFormer -mti 200

@REM call python .\test_cases\test_robustness.py -d .\datasets\test_suite_robustness\SynthDet_robustness_groceries\ -chdir checkpoints_robustness_groceries_TF -dc groceries -c .\configs\SynthDet\SynthDet_robustness_test_pretrained_TF.py -mn rgbd rgbd_variation -m TokenFusion -mti 200

@REM call python .\test_cases\test_robustness.py -d .\datasets\test_suite_robustness\SynthDet_robustness_groceries\ -chdir checkpoints_robustness_groceries_SF -dc groceries -c .\configs\SynthDet\SynthDet_robustness_test_pretrained_SF.py -mn rgb rgb_variation -m SegFormer -mti 200

@REM call python .\test_cases\test_robustness.py -d .\datasets\test_suite_robustness\SynthDet_robustness_groceries\ -chdir checkpoints_robustness_groceries_CMX -dc groceries -c .\configs\SynthDet\SynthDet_robustness_test_pretrained_CMX.py -mn rgbd rgbd_variation -m CMX -mti 200


@REM call python .\test_cases\test_fgbg_aux.py -d .\datasets\test_suite_fgbg_large_spacing_appearance\ -chdir checkpoints_fgbg_spacing -c .\configs\SynthDet\SynthDet_foreground_background_test_pretrained_aux.py -mti 200 -mn rgbd rgbd_aux -m DFormer

@REM call python .\test_cases\test_fgbg_aux.py -d .\datasets\test_suite_fgbg_large_spacing_appearance\ -chdir checkpoints_fgbg_spacing_TF -c .\configs\SynthDet\SynthDet_foreground_background_test_pretrained_TF.py -mti 200 -m TokenFusion -mn rgbd

@REM call python .\test_cases\test_fgbg_aux.py -d .\datasets\test_suite_fgbg_large_spacing_appearance\ -chdir checkpoints_fgbg_spacing_SF -c .\configs\SynthDet\SynthDet_foreground_background_test_pretrained_SF.py -mti 200 -m SegFormer -mn rgb

@REM call python .\test_cases\test_fgbg_aux.py -d .\datasets\test_suite_fgbg_large_spacing_appearance\ -chdir checkpoints_fgbg_spacing_CMX -c .\configs\SynthDet\SynthDet_foreground_background_test_pretrained_CMX.py -mti 200 -m CMX -mn rgbd



@REM call python .\test_cases\test_spatial.py -d .\datasets\test_suite_spatial_realistic_no_walls\ -chdir checkpoints_spatial_groceries -dc groceries -c .\configs\SynthDet\SynthDet_spatial_test_pretrained.py -mti 200 -mn rgbd -m DFormer

@REM call python .\test_cases\test_spatial.py -d .\datasets\test_suite_spatial_realistic_no_walls\ -chdir checkpoints_spatial_groceries_TF -dc groceries -c .\configs\SynthDet\SynthDet_spatial_test_pretrained_TF.py -mti 200 -mn rgbd -m TokenFusion

@REM call python .\test_cases\test_spatial.py -d .\datasets\test_suite_spatial_realistic_no_walls\ -chdir checkpoints_spatial_groceries_SF -dc groceries -c .\configs\SynthDet\SynthDet_spatial_test_pretrained_SF.py -mti 200 -mn rgb -m SegFormer

@REM call python .\test_cases\test_spatial.py -d .\datasets\test_suite_spatial_realistic_no_walls\ -chdir checkpoints_spatial_groceries_CMX -dc groceries -c .\configs\SynthDet\SynthDet_spatial_test_pretrained_CMX.py -mti 200 -mn rgbd -m CMX


@REM Remove if not needed
@REM shutdown /h