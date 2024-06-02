@REM call .\convert_and_train.bat . rgb_noise gems16 60 5 DFormer-Tiny

@REM ren UsefullnessOfDepth\datasets\SynthDet_rgb_noise\Depth Depth_original
@REM ren UsefullnessOfDepth\datasets\SynthDet_rgb_noise\Depth_noise Depth

@REM ren UsefullnessOfDepth\datasets\SynthDet_rgb_noise\RGB RGB_noise
@REM ren UsefullnessOfDepth\datasets\SynthDet_rgb_noise\RGB_original RGB

@REM call .\convert_and_train.bat . rgb_noise gems16 60 5 DFormer-Tiny

@REM spatial_large_path =        r"..\datasets\spatial\SynthDet_groceries_spatial_realistic_large"
@REM spatial_medium_path =       r"..\datasets\spatial\SynthDet_groceries_spatial_realistic_medium"
@REM spatial_small_close_path =  r"..\datasets\spatial\SynthDet_groceries_spatial_realistic_close"
@REM spatial_small_far_path =    r"..\datasets\spatial\SynthDet_groceries_spatial_realistic_far"
@REM spatial_small_avg_path =    r"..\datasets\spatial\SynthDet_groceries_spatial_realistic_half"

call create_dataset.bat C:\Users\Pasca\AppData\LocalLow\UnityTechnologies\SynthDet\solo_191 fgbg_orange_flat groceries
call create_dataset.bat C:\Users\Pasca\AppData\LocalLow\UnityTechnologies\SynthDet\solo_192 fgbg_orange_object groceries
call create_dataset.bat C:\Users\Pasca\AppData\LocalLow\UnityTechnologies\SynthDet\solo_194 fgbg_orange_texture groceries
call create_dataset.bat C:\Users\Pasca\AppData\LocalLow\UnityTechnologies\SynthDet\solo_193 fgbg_orange_object_texture groceries

call create_dataset.bat C:\Users\Pasca\AppData\LocalLow\UnityTechnologies\SynthDet\solo_195 fgbg_green_flat groceries
call create_dataset.bat C:\Users\Pasca\AppData\LocalLow\UnityTechnologies\SynthDet\solo_196 fgbg_green_texture groceries
call create_dataset.bat C:\Users\Pasca\AppData\LocalLow\UnityTechnologies\SynthDet\solo_197 fgbg_green_object groceries
call create_dataset.bat C:\Users\Pasca\AppData\LocalLow\UnityTechnologies\SynthDet\solo_198 fgbg_green_object_texture groceries

@REM call convert_and_train.bat . groceries_spatial_realistic_large groceries8 60 5 DFormer-Tiny

@REM cd ..

@REM call convert_and_train.bat . groceries_spatial_realistic_medium groceries8 60 5 DFormer-Tiny

@REM cd ..

@REM call convert_and_train.bat . groceries_spatial_realistic_half groceries8 60 -1 DFormer-Tiny

@REM cd ..

@REM call convert_and_train.bat . groceries_spatial_realistic_close groceries8 60 5 DFormer-Tiny

@REM cd ..

@REM call convert_and_train.bat . groceries_spatial_realistic_far groceries8 60 5 DFormer-Tiny

@REM cd ..

@REM set dataset_names=(SynthDet_groceries_spatial_realistic_medium SynthDet_groceries_spatial_realistic_large SynthDet_groceries_spatial_realistic_close SynthDet_groceries_spatial_realistic_half SynthDet_groceries_spatial_realistic_far)

@REM cd UsefullnessOfDepth
@REM for %%d in %dataset_names% do (
@REM     @REM @REM code\UsefullnessOfDepth\checkpoints\SynthDet_groceries_spatial_small_depth_far_DFormer-Tiny\run_20240509-164731\epoch_50_miou_86.44.pth
@REM     @REM python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_groceries_spatial_small_depth_far_DFormer --model_weights checkpoints\SynthDet_groceries_spatial_small_depth_far_DFormer-Tiny\run_20240509-164731\epoch_50_miou_86.44.pth --bin_size 1000 --model DFormer-Tiny --dataset datasets\%%d

@REM     @REM code\UsefullnessOfDepth\checkpoints\SynthDet_groceries_spatial_realistic_close_DFormer-Tiny\run_20240516-181859\epoch_60_miou_84.607.pth
@REM     call python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_groceries_spatial_realistic_close_DFormer --model_weights checkpoints\SynthDet_groceries_spatial_realistic_close_DFormer-Tiny\run_20240516-181859\epoch_60_miou_84.607.pth --bin_size 1000 --model DFormer-Tiny --dataset datasets\spatial\%%d

@REM     @REM code\UsefullnessOfDepth\checkpoints\SynthDet_groceries_spatial_realistic_far_DFormer-Tiny\run_20240516-190940\epoch_60_miou_83.978.pth
@REM     call python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_groceries_spatial_realistic_far_DFormer --model_weights checkpoints\SynthDet_groceries_spatial_realistic_far_DFormer-Tiny\run_20240516-190940\epoch_60_miou_83.978.pth --bin_size 1000 --model DFormer-Tiny --dataset datasets\spatial\%%d

@REM     @REM code\UsefullnessOfDepth\checkpoints\SynthDet_groceries_spatial_realistic_half_DFormer-Tiny\run_20240516-173236\epoch_60_miou_85.592.pth
@REM     call python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_groceries_spatial_realistic_half_DFormer --model_weights checkpoints\SynthDet_groceries_spatial_realistic_half_DFormer-Tiny\run_20240516-173236\epoch_60_miou_85.592.pth --bin_size 1000 --model DFormer-Tiny --dataset datasets\spatial\%%d

@REM     @REM code\UsefullnessOfDepth\checkpoints\SynthDet_groceries_spatial_realistic_large_DFormer-Tiny\run_20240516-154521\epoch_60_miou_82.742.pth
@REM     call python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_groceries_spatial_realistic_large_DFormer --model_weights checkpoints\SynthDet_groceries_spatial_realistic_large_DFormer-Tiny\run_20240516-154521\epoch_60_miou_82.742.pth --bin_size 1000 --model DFormer-Tiny --dataset datasets\spatial\%%d

@REM     @REM code\UsefullnessOfDepth\checkpoints\SynthDet_groceries_spatial_realistic_medium_DFormer-Tiny\run_20240516-163336\epoch_60_miou_85.026.pth
@REM     call python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_groceries_spatial_realistic_medium_DFormer --model_weights checkpoints\SynthDet_groceries_spatial_realistic_medium_DFormer-Tiny\run_20240516-163336\epoch_60_miou_85.026.pth --bin_size 1000 --model DFormer-Tiny --dataset datasets\spatial\%%d
@REM )


@REM To Depth
@REM And move RGB to RGB_noise
@REM And move RGB_original to RGB


@REM call convert_and_train.bat . groceries_spatial_large_depth groceries8 60 5 DFormer-Tiny

@REM cd ..

@REM call convert_and_train.bat . groceries_spatial_medium_depth groceries8 60 5 DFormer-Tiny

@REM cd ..

@REM call convert_and_train.bat . groceries_spatial_small_depth_avg groceries8 60 5 DFormer-Tiny

@REM set dataset_names=("groceries_spatial_large_depth" "groceries_spatial_medium_depth" "groceries_spatial_small_depth_avg" "groceries_spatial_small_depth_close" "groceries_spatial_small_depth_far")

@REM for %%dataset in %dataset_names% do (
@REM     call python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_groceries_spatial_small_depth_avg_DFormer --model_weights checkpoints\SynthDet_groceries_spatial_small_depth_avg_DFormer-Tiny\run_20240510-014658\epoch_60_miou_86.021.pth --bin_size 1000 --model DFormer-Tiny --dataset datasets\$dataset

@REM     @REM code\UsefullnessOfDepth\checkpoints\SynthDet_groceries_spatial_medium_depth_DFormer-Tiny\run_20240510-010329\epoch_60_miou_83.396.pth
@REM     call python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_groceries_spatial_medium_depth_DFormer --model_weights checkpoints\SynthDet_groceries_spatial_medium_depth_DFormer-Tiny\run_20240510-010329\epoch_60_miou_83.396.pth --bin_size 1000 --model DFormer-Tiny --dataset datasets\$dataset

@REM     @REM code\UsefullnessOfDepth\checkpoints\SynthDet_groceries_spatial_large_depth_DFormer-Tiny\run_20240510-002246\epoch_60_miou_84.673.pth
@REM     call python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_groceries_spatial_large_depth_DFormer --model_weights checkpoints\SynthDet_groceries_spatial_large_depth_DFormer-Tiny\run_20240510-002246\epoch_60_miou_84.673.pth --bin_size 1000 --model DFormer-Tiny --dataset datasets\$dataset
@REM )

@echo off

@REM @REM set dataset_names=(groceries_spatial_large_depth groceries_spatial_medium_depth groceries_spatial_small_depth_avg groceries_spatial_small_depth_close groceries_spatial_small_depth_far)
@REM set dataset_names=(groceries_spatial_large_depth groceries_spatial_medium_depth)

@REM cd UsefullnessOfDepth
@REM for %%d in %dataset_names% do (
@REM     @REM python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_groceries_spatial_small_depth_avg_DFormer --model_weights checkpoints\SynthDet_groceries_spatial_small_depth_avg_DFormer-Tiny\run_20240510-014658\epoch_60_miou_86.021.pth --bin_size 1000 --model DFormer-Tiny --dataset datasets\SynthDet_%%d

@REM     @REM python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_groceries_spatial_medium_depth_DFormer --model_weights checkpoints\SynthDet_groceries_spatial_medium_depth_DFormer-Tiny\run_20240510-010329\epoch_60_miou_83.396.pth --bin_size 1000 --model DFormer-Tiny --dataset datasets\SynthDet_%%d

@REM     @REM python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_groceries_spatial_large_depth_DFormer --model_weights checkpoints\SynthDet_groceries_spatial_large_depth_DFormer-Tiny\run_20240510-002246\epoch_60_miou_84.673.pth --bin_size 1000 --model DFormer-Tiny --dataset datasets\SynthDet_%%d

@REM     @REM code\UsefullnessOfDepth\checkpoints\SynthDet_groceries_spatial_small_depth_close_DFormer-Tiny\run_20240508-225242\epoch_60_miou_85.807.pth
@REM     python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_groceries_spatial_small_depth_DFormer --model_weights checkpoints\SynthDet_groceries_spatial_small_depth_close_DFormer-Tiny\run_20240508-225242\epoch_60_miou_85.807.pth --bin_size 1000 --model DFormer-Tiny --dataset datasets\SynthDet_%%d

@REM     @REM code\UsefullnessOfDepth\checkpoints\SynthDet_groceries_spatial_small_depth_far_DFormer-Tiny\run_20240509-164731\epoch_50_miou_86.44.pth
@REM     python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_groceries_spatial_small_depth_far_DFormer --model_weights checkpoints\SynthDet_groceries_spatial_small_depth_far_DFormer-Tiny\run_20240509-164731\epoch_50_miou_86.44.pth --bin_size 1000 --model DFormer-Tiny --dataset datasets\SynthDet_%%d
@REM )
cd ..


@REM Remove if not needed
@REM shutdown /h