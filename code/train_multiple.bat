
cd UsefullnessOfDepth

@REM call python .\test_cases\test_robustness.py -d .\datasets\test_suite_robustness\SynthDet_robustness_groceries\ -chdir checkpoints_robustness_groceries -dc groceries -c .\configs\SynthDet\SynthDet_robustness_test_pretrained.py -mn rgbd rgbd_variation depth -m DFormer -mti 100

@REM call python .\test_cases\test_robustness.py -d .\datasets\test_suite_robustness\SynthDet_robustness_groceries\ -chdir checkpoints_robustness_groceries_non_rot_TF -dc groceries -c .\configs\SynthDet\SynthDet_robustness_test_pretrained_TF.py -mn rgbd rgbd_variation -m TokenFusion -mti 100

@REM call python .\test_cases\test_robustness.py -d .\datasets\test_suite_robustness\SynthDet_robustness_groceries\ -chdir checkpoints_robustness_groceries_SF -dc groceries -c .\configs\SynthDet\SynthDet_robustness_test_pretrained_SF.py -mn rgb rgb_variation -m SegFormer -mti 100

@REM call python .\test_cases\test_robustness.py -d .\datasets\test_suite_robustness\SynthDet_robustness_groceries\ -chdir checkpoints_robustness_groceries_CMX -dc groceries -c .\configs\SynthDet\SynthDet_robustness_test_pretrained_CMX.py -mn rgbd rgbd_variation -m CMX -mti 100


@REM call python .\test_cases\test_fgbg_aux.py -d .\datasets\test_suite_fgbg_large_spacing_appearance\ -chdir checkpoints_fgbg_spacing -c .\configs\SynthDet\SynthDet_foreground_background_test_pretrained_aux.py -mti 100 -mn rgbd rgbd_aux -m DFormer -l D:\mscPascalBenschop\code\UsefullnessOfDepth\checkpoints_fgbg_spacing\log_20240609_093636.txt

@REM call python .\test_cases\test_fgbg_aux.py -d .\datasets\test_suite_fgbg_large_spacing_appearance\ -chdir checkpoints_fgbg_spacing_TF -c .\configs\SynthDet\SynthDet_foreground_background_test_pretrained_TF.py -mti 100 -m TokenFusion -mn rgbd -l D:\mscPascalBenschop\code\UsefullnessOfDepth\checkpoints_fgbg_spacing_TF\log_20240609_110952.txt

@REM call python .\test_cases\test_fgbg_aux.py -d .\datasets\test_suite_fgbg_large_spacing_appearance\ -chdir checkpoints_fgbg_spacing_SF -c .\configs\SynthDet\SynthDet_foreground_background_test_pretrained_SF.py -mti 100 -m SegFormer -mn rgb -l D:\mscPascalBenschop\code\UsefullnessOfDepth\checkpoints_fgbg_spacing_SF\log_20240609_112910.txt

@REM call python .\test_cases\test_fgbg_aux.py -d .\datasets\test_suite_fgbg_large_spacing_appearance\ -chdir checkpoints_fgbg_spacing_CMX -c .\configs\SynthDet\SynthDet_foreground_background_test_pretrained_CMX.py -mti 100 -m CMX -mn rgbd -l D:\mscPascalBenschop\code\UsefullnessOfDepth\checkpoints_fgbg_spacing_CMX\log_20240609_114140.txt


call python .\test_cases\test_fgbg_aux.py -d .\datasets\test_suite_fgbg_large_spacing_appearance\ -chdir checkpoints_fgbg_spacing -c .\configs\SynthDet\SynthDet_foreground_background_test_pretrained_aux_S.py -mti 200 -mn rgbd rgbd_aux -m DFormer

call python .\test_cases\test_fgbg_aux.py -d .\datasets\test_suite_fgbg_large_spacing_appearance\ -chdir checkpoints_fgbg_spacing_TF -c .\configs\SynthDet\SynthDet_foreground_background_test_pretrained_TF.py -mti 200 -m TokenFusion -mn rgbd

call python .\test_cases\test_fgbg_aux.py -d .\datasets\test_suite_fgbg_large_spacing_appearance\ -chdir checkpoints_fgbg_spacing_SF -c .\configs\SynthDet\SynthDet_foreground_background_test_pretrained_SF.py -mti 200 -m SegFormer -mn rgb

call python .\test_cases\test_fgbg_aux.py -d .\datasets\test_suite_fgbg_large_spacing_appearance\ -chdir checkpoints_fgbg_spacing_CMX -c .\configs\SynthDet\SynthDet_foreground_background_test_pretrained_CMX.py -mti 200 -m CMX -mn rgbd


@REM call python .\test_cases\test_spatial.py -d .\datasets\test_suite_spatial_realistic_no_walls\ -chdir checkpoints_spatial_groceries -dc groceries -c .\configs\SynthDet\SynthDet_spatial_test_pretrained.py -mti 200 -mn rgbd -m DFormer -l G:\mscPascalBenschop\code\UsefullnessOfDepth\checkpoints_spatial_groceries\log_20240611_111159.txt

@REM call python .\test_cases\test_spatial.py -d .\datasets\test_suite_spatial_realistic_no_walls\ -chdir checkpoints_spatial_groceries_TF -dc groceries -c .\configs\SynthDet\SynthDet_spatial_test_pretrained_TF.py -mti 200 -mn rgbd -m TokenFusion -l G:\mscPascalBenschop\code\UsefullnessOfDepth\checkpoints_spatial_groceries_TF\log_20240611_130207.txt

@REM call python .\test_cases\test_spatial.py -d .\datasets\test_suite_spatial_realistic_no_walls\ -chdir checkpoints_spatial_groceries_SF -dc groceries -c .\configs\SynthDet\SynthDet_spatial_test_pretrained_SF.py -mti 200 -mn rgb -m SegFormer -l G:\mscPascalBenschop\code\UsefullnessOfDepth\checkpoints_spatial_groceries_SF\log_20240611_153044.txt

@REM call python .\test_cases\test_spatial.py -d .\datasets\test_suite_spatial_realistic_no_walls\ -chdir checkpoints_spatial_groceries_CMX -dc groceries -c .\configs\SynthDet\SynthDet_spatial_test_pretrained_CMX.py -mti 200 -mn rgbd -m CMX -l G:\mscPascalBenschop\code\UsefullnessOfDepth\checkpoints_spatial_groceries_CMX\log_20240611_164812.txt


@REM Remove if not needed
@REM shutdown /h