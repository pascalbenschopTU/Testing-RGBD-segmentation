REM This script is used to convert the dataset from SOLO format to COCO format and then to DFormer format
set dataset_location=%1
set dataset_name=%2
set use_gems=%3

set use_edge_enhancement=false

set checkpoint_dir=checkpoints

set dataset_path=..\DFormer\datasets\SUNRGBD
set config_path=local_configs.SUNRGBD.SUNRGBD_DFormer_Tiny

IF "%use_gems%"=="" (
    set use_gems=false
)

IF %use_edge_enhancement% == true (
    python add_edge_enhancement.py %dataset_path%
)

cd DFormer

@REM Train the model
@REM python utils\train.py --config=%config_path% --gpus 1 --checkpoint_dir %checkpoint_dir%

@REM Create prediction mious for the dataset with RGB-Depth

code\DFormer\checkpoints\SUNRGBD_DFormer-Tiny\run_20240329-223003

REM Get the last directory starting with "run" from the given path
for /f "delims=" %%d in ('dir /b /ad /on %checkpoint_dir%\SUNRGBD_DFormer-Tiny\run*') do (
    set "last_directory=%%d"
)

REM Get the last filename starting with "epoch" in the last directory
for /f "delims=" %%f in ('dir /b /a-d /on %checkpoint_dir%\SUNRGBD_DFormer-Tiny\%last_directory%\epoch*') do (
    set "last_filename=%%f"
)

echo Last directory: %last_directory%
echo Last filename: %last_filename%

set rgb_depth_model_weights=%checkpoint_dir%\SUNRGBD_DFormer-Tiny\%last_directory%\%last_filename%

python utils\evaluate_models.py --config=%config_path% --model_weights %rgb_depth_model_weights%

move "%dataset_path%\Depth" "%dataset_path%\Depth_original"
move "%dataset_path%\Grayscale" "%dataset_path%\Depth"


@REM Train the model
python utils\train.py --config=%config_path% --gpus 1 --checkpoint_dir %checkpoint_dir%

REM Get the last directory starting with "run" from the given path
for /f "delims=" %%d in ('dir /b /ad /on %checkpoint_dir%\SUNRGBD_DFormer-Tiny\run*') do (
    set "last_directory=%%d"
)

REM Get the last filename starting with "epoch" in the last directory
for /f "delims=" %%f in ('dir /b /a-d /on %checkpoint_dir%\SUNRGBD_DFormer-Tiny\%last_directory%\epoch*') do (
    set "last_filename=%%f"
)

echo Last directory: %last_directory%
echo Last filename: %last_filename%

set rgb_black_model_weights=%checkpoint_dir%\SUNRGBD_DFormer-Tiny\%last_directory%\%last_filename%

python utils\evaluate_models.py --config=%config_path% --model_weights %rgb_black_model_weights%


@REM Evaluate the models
python utils\create_predictions.py --model_a_path %rgb_black_model_weights% --model_b_path %rgb_depth_model_weights% --dir_dataset datasets\SUNRGBD --config=%config_path%

@REM Create gen_results.txt and store the last command
echo utils\create_predictions.py --model_a_path %rgb_black_model_weights% --model_b_path %rgb_depth_model_weights% --dir_dataset datasets\SUNRGBD --config=%config_path% >> %checkpoint_dir%\SUNRGBD_DFormer-Tiny\gen_results.txt
