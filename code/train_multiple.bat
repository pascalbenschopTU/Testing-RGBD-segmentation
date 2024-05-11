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

@REM set dataset_names=(groceries_spatial_large_depth groceries_spatial_medium_depth groceries_spatial_small_depth_avg groceries_spatial_small_depth_close groceries_spatial_small_depth_far)
set dataset_names=(groceries_spatial_large_depth groceries_spatial_medium_depth)

cd UsefullnessOfDepth
for %%d in %dataset_names% do (
    @REM python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_groceries_spatial_small_depth_avg_DFormer --model_weights checkpoints\SynthDet_groceries_spatial_small_depth_avg_DFormer-Tiny\run_20240510-014658\epoch_60_miou_86.021.pth --bin_size 1000 --model DFormer-Tiny --dataset datasets\SynthDet_%%d

    @REM python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_groceries_spatial_medium_depth_DFormer --model_weights checkpoints\SynthDet_groceries_spatial_medium_depth_DFormer-Tiny\run_20240510-010329\epoch_60_miou_83.396.pth --bin_size 1000 --model DFormer-Tiny --dataset datasets\SynthDet_%%d

    @REM python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_groceries_spatial_large_depth_DFormer --model_weights checkpoints\SynthDet_groceries_spatial_large_depth_DFormer-Tiny\run_20240510-002246\epoch_60_miou_84.673.pth --bin_size 1000 --model DFormer-Tiny --dataset datasets\SynthDet_%%d

    @REM code\UsefullnessOfDepth\checkpoints\SynthDet_groceries_spatial_small_depth_close_DFormer-Tiny\run_20240508-225242\epoch_60_miou_85.807.pth
    python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_groceries_spatial_small_depth_DFormer --model_weights checkpoints\SynthDet_groceries_spatial_small_depth_close_DFormer-Tiny\run_20240508-225242\epoch_60_miou_85.807.pth --bin_size 1000 --model DFormer-Tiny --dataset datasets\SynthDet_%%d

    @REM code\UsefullnessOfDepth\checkpoints\SynthDet_groceries_spatial_small_depth_far_DFormer-Tiny\run_20240509-164731\epoch_50_miou_86.44.pth
    python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_groceries_spatial_small_depth_far_DFormer --model_weights checkpoints\SynthDet_groceries_spatial_small_depth_far_DFormer-Tiny\run_20240509-164731\epoch_50_miou_86.44.pth --bin_size 1000 --model DFormer-Tiny --dataset datasets\SynthDet_%%d
)
cd ..


@REM Remove if not needed
@REM shutdown /h