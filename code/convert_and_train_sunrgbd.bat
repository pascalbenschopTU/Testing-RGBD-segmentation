REM This script is used to convert the dataset from SOLO format to COCO format and then to DFormer format
set dataset_type=SUNRGBD

set checkpoint_dir=checkpoints

set dataset_path=..\DFormer\datasets\SUNRGBD
set config_path=local_configs.SUNRGBD.SUNRGBD_DFormer_Tiny

cd DFormer

@REM Train the model
@REM python utils\train_clean.py --config=%config_path% --gpus 1 --checkpoint_dir %checkpoint_dir% --dataset_type %dataset_type% --num_hyperparameter_epochs 0 --x_channels=3 --x_e_channels=1

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

@REM python utils\evaluate_models.py --config=%config_path% --model_weights %rgb_depth_model_weights%

@REM move "%dataset_path%\Depth" "%dataset_path%\Depth_original"
@REM xcopy "%dataset_path%\RGB" "%dataset_path%\Depth" /E /I /Y

@REM Train the model
python utils\train_clean.py --config=%config_path% --gpus 1 --checkpoint_dir %checkpoint_dir% --dataset_type %dataset_type% --num_hyperparameter_epochs 0 --x_channels=3 --x_e_channels=3

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

set rgb_model_weights=%checkpoint_dir%\SUNRGBD_DFormer-Tiny\%last_directory%\%last_filename%

python utils\evaluate_models.py --config=%config_path% --model_weights %rgb_model_weights%


@REM Evaluate the models
python utils\create_predictions.py --model_a_path %rgb_model_weights% --model_b_path %rgb_depth_model_weights% --dir_dataset datasets\SUNRGBD --config=%config_path% --test_mode

@REM Create gen_results.txt and store the last command
echo utils\create_predictions.py --model_a_path %rgb_model_weights% --model_b_path %rgb_depth_model_weights% --dir_dataset datasets\SUNRGBD --config=%config_path% >> %checkpoint_dir%\SUNRGBD_DFormer-Tiny\gen_results.txt


@REM ####################################### DEPTH ONLY #######################################


move "%dataset_path%\Depth" "%dataset_path%\RGB_copy"
move "%dataset_path%\Depth_original" "%dataset_path%\Depth"
move "%dataset_path%\RGB" "%dataset_path%\RGB_original"
@REM Copy the directory and its files
xcopy "%dataset_path%\Depth" "%dataset_path%\RGB" /E /I /Y


@REM Train the model
python utils\train_clean.py --config=%config_path% --gpus 1 --checkpoint_dir %checkpoint_dir% --dataset_type %dataset_type% --num_hyperparameter_epochs 0 --x_channels=1 --x_e_channels=1

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

set depth_only_model_weights=%checkpoint_dir%\SUNRGBD_DFormer-Tiny\%last_directory%\%last_filename%

python utils\evaluate_models.py --config=%config_path% --model_weights %depth_only_model_weights%
