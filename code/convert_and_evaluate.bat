@echo off
REM This script is used to convert the dataset from SOLO format to COCO format and then to DFormer format

set dataset_folder=%1
set use_gems=%2
set use_cars=%3

set checkpoint_dir=checkpoints
set test_dataset_dir=test_dataset_dir

IF "%use_gems%"=="" set use_gems=false
IF "%use_cars%"=="" set use_cars=false

@REM echo use_cars is: %use_cars%

for /f "delims=" %%d in ('dir /b /ad "%dataset_folder%\*"') do (
    call :process_dataset "%%d"
)
goto :eof

:process_dataset
setlocal enabledelayedexpansion
set dataset_name=%~1
set dataset_location=!dataset_folder!\!dataset_name!

echo Converting dataset !dataset_name! from SOLO format to COCO format

REM Creating dataset
solo2coco !dataset_location! data\SynthDet\

cd SynthDet_Pascal
python convert_solo_depth_to_coco.py !dataset_location! ..\data\SynthDet\coco\depth

REM Rename the dataset to the correct name
move "..\data\SynthDet\coco" "..\data\SynthDet\coco_!dataset_name!"

set dataset_path=..\data\SynthDet\coco_!dataset_name!

IF "%use_gems%" == "true" (
    REM Convert the dataset to DFormer format with gems
    python convert_coco_to_dformer.py !dataset_path! ..\DFormer\!test_dataset_dir!\SynthDet_!dataset_name! --gems --test_mode
) ELSE (
    IF "%use_cars%" == "true" (
        REM Convert the dataset to DFormer format with cars
        python convert_coco_to_dformer.py !dataset_path! ..\DFormer\!test_dataset_dir!\SynthDet_!dataset_name! --cars --test_mode
    ) ELSE (
        REM Convert the dataset to DFormer format
        python convert_coco_to_dformer.py !dataset_path! ..\DFormer\!test_dataset_dir!\SynthDet_!dataset_name! --test_mode
    )
)

REM Remove the dataset from the data folder
rmdir /s /q !dataset_path!

REM Change to the DFormer directory
cd /d ..\DFormer

REM Get the first directory starting with "run" from the given path
for /f "delims=" %%d in ('dir /b /ad /on !checkpoint_dir!\SynthDet_!dataset_name!_DFormer-Tiny\run*') do (
    set "last_directory=%%d"
    goto :continue
)

:continue
REM Get the last directory starting with "run" from the given path

REM Get the last filename starting with "epoch" in the last directory
for /f "delims=" %%f in ('dir /b /a-d /on !checkpoint_dir!\SynthDet_!dataset_name!_DFormer-Tiny\!last_directory!\epoch*') do (
    set "last_filename=%%f"
)

echo Last directory: !last_directory!
echo Last filename: !last_filename!

set rgb_depth_model_weights=!checkpoint_dir!\SynthDet_!dataset_name!_DFormer-Tiny\!last_directory!\!last_filename!

python utils\evaluate_models.py --config=local_configs.SynthDet.SynthDet_!dataset_name!_Dformer_Tiny --model_weights !rgb_depth_model_weights! --dataset !test_dataset_dir!

set new_dataset_path=..\DFormer\!test_dataset_dir!\SynthDet_!dataset_name!
move "!new_dataset_path!\Depth" "!new_dataset_path!\Depth_original"
move "!new_dataset_path!\Grayscale" "!new_dataset_path!\Depth"

REM Get the last directory starting with "run" from the given path
set count=0
for /f "delims=" %%d in ('dir /b /ad /on !checkpoint_dir!\SynthDet_!dataset_name!_DFormer-Tiny\run*') do (
    set "last_directory=%%d"
    set /a count+=1
    if !count! == 2 (
        goto :break_loop
    )
)
:break_loop
REM Get the last filename starting with "epoch" in the last directory

REM Get the last filename starting with "epoch" in the last directory
for /f "delims=" %%f in ('dir /b /a-d /on !checkpoint_dir!\SynthDet_!dataset_name!_DFormer-Tiny\!last_directory!\epoch*') do (
    set "last_filename=%%f"
)

echo Last directory: !last_directory!
echo Last filename: !last_filename!

set rgb_black_model_weights=!checkpoint_dir!\SynthDet_!dataset_name!_DFormer-Tiny\!last_directory!\!last_filename!

python utils\evaluate_models.py --config=local_configs.SynthDet.SynthDet_!dataset_name!_Dformer_Tiny --model_weights !rgb_black_model_weights! --dataset !test_dataset_dir!

REM Evaluate the models
python utils\create_predictions.py --model_a_path !rgb_black_model_weights! --model_b_path !rgb_depth_model_weights! --dir_dataset !test_dataset_dir!\SynthDet_!dataset_name! --config=local_configs.SynthDet.SynthDet_!dataset_name!_Dformer_Tiny --test_mode

REM Create gen_results.txt and store the last command
echo utils\create_predictions.py --model_a_path !rgb_black_model_weights! --model_b_path !rgb_depth_model_weights! --dir_dataset !test_dataset_dir!\SynthDet_!dataset_name! --config=local_configs.SynthDet.SynthDet_!dataset_name!_Dformer_Tiny --test_mode > !test_dataset_dir!\SynthDet_!dataset_name!\gen_results.txt

echo Done with dataset !dataset_name!
goto :eof

echo Done.
