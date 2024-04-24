REM This script is used to convert the dataset from SOLO format to COCO format and then to UsefullnessOfDepth format
set dataset_location=%1
set dataset_name=%2
set dataset_type=%3
set num_epochs=%4
set hyperparam_epochs=%5

set use_edge_enhancement=false
set checkpoint_dir=checkpoints
@REM set checkpoint_dir=checkpoints_CMX
set bin_size=25

IF "%num_epochs%" == "" (
    set num_epochs=60
)

IF "%hyperparam_epochs%" == "" (
    set hyperparam_epochs=5
)

IF EXIST "UsefullnessOfDepth\datasets\SynthDet_%dataset_name%" goto :skip_dataset_creation

@REM Creating dataset
solo2coco %dataset_location% data\SynthDet\

cd UsefullnessOfDepth/SynthDet_tools
python convert_solo_depth_to_coco.py %dataset_location% ..\..\data\SynthDet\coco\depth

@REM Rename the dataset to the correct name
move "..\..\data\SynthDet\coco" "..\..\data\SynthDet\coco_%dataset_name%"

set dataset_path=..\..\data\SynthDet\coco_%dataset_name%

python convert_coco_to_dformer.py %dataset_path% ..\datasets\SynthDet_%dataset_name% %dataset_type%

IF %use_edge_enhancement% == true (
    python add_edge_enhancement.py ..\datasets\SynthDet_%dataset_name%
)

@REM Remove the dataset from the data folder
rmdir /s /q %dataset_path%

@REM Change to the UsefullnessOfDepth directory
cd ../..

:skip_dataset_creation
cd UsefullnessOfDepth

@REM Copy the template file to the new file
copy configs\SynthDet\SynthDet_template_DFormer_Tiny.py configs\SynthDet\SynthDet_%dataset_name%_DFormer_Tiny.py

@REM Train the model
@REM python utils\train_clean.py --config=configs.SynthDet.SynthDet_%dataset_name%_DFormer_Tiny --gpus 1 --checkpoint_dir %checkpoint_dir% --dataset_type %dataset_type% --x_channels=3 --x_e_channels=1 --num_epochs %num_epochs% --num_hyperparameter_epochs %hyperparam_epochs%
python utils\train.py ^
--config=configs.SynthDet.SynthDet_%dataset_name%_DFormer_Tiny ^
--gpus 1 ^
--checkpoint_dir %checkpoint_dir% ^
--dataset_type %dataset_type% ^
--x_channels=3 ^
--x_e_channels=1 ^
--num_epochs %num_epochs% ^
--num_hyperparameter_epochs %hyperparam_epochs%


REM Get the last directory starting with "run" from the given path
for /f "delims=" %%d in ('dir /b /ad /on %checkpoint_dir%\SynthDet_%dataset_name%_DFormer-Tiny\run*') do (
    set "last_directory=%%d"
)

REM Get the last filename starting with "epoch" in the last directory
for /f "delims=" %%f in ('dir /b /a-d /on %checkpoint_dir%\SynthDet_%dataset_name%_DFormer-Tiny\%last_directory%\epoch*') do (
    set "last_filename=%%f"
)

echo Last directory: %last_directory%
echo Last filename: %last_filename%

set rgb_depth_model_weights=%checkpoint_dir%\SynthDet_%dataset_name%_DFormer-Tiny\%last_directory%\%last_filename%

python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_%dataset_name%_DFormer_Tiny --model_weights %rgb_depth_model_weights% --bin_size %bin_size%

set new_dataset_path=..\UsefullnessOfDepth\datasets\SynthDet_%dataset_name%
move "%new_dataset_path%\Depth" "%new_dataset_path%\Depth_original"
xcopy "%new_dataset_path%\RGB" "%new_dataset_path%\Depth" /E /I /Y


@REM Train the model
python utils\train.py ^
--config=configs.SynthDet.SynthDet_%dataset_name%_DFormer_Tiny ^
--gpus 1 ^
--checkpoint_dir %checkpoint_dir% ^
--dataset_type %dataset_type% ^
--x_channels=3 ^
--x_e_channels=3 ^
--num_epochs %num_epochs% ^
--num_hyperparameter_epochs %hyperparam_epochs%

REM Get the last directory starting with "run" from the given path
for /f "delims=" %%d in ('dir /b /ad /on %checkpoint_dir%\SynthDet_%dataset_name%_DFormer-Tiny\run*') do (
    set "last_directory=%%d"
)

REM Get the last filename starting with "epoch" in the last directory
for /f "delims=" %%f in ('dir /b /a-d /on %checkpoint_dir%\SynthDet_%dataset_name%_DFormer-Tiny\%last_directory%\epoch*') do (
    set "last_filename=%%f"
)

echo Last directory: %last_directory%
echo Last filename: %last_filename%

set rgb_only_model_weights=%checkpoint_dir%\SynthDet_%dataset_name%_DFormer-Tiny\%last_directory%\%last_filename%

python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_%dataset_name%_DFormer_Tiny --model_weights %rgb_only_model_weights% --bin_size %bin_size%


@REM Create gen_results.txt and store the last command
echo utils\create_predictions.py --model_a_path %rgb_only_model_weights% --model_b_path %rgb_depth_model_weights% --dir_dataset datasets\SynthDet_%dataset_name% --config=configs.SynthDet.SynthDet_%dataset_name%_DFormer_Tiny >> %checkpoint_dir%\SynthDet_%dataset_name%_DFormer-Tiny\gen_results.txt


@REM ####################################### DEPTH ONLY #######################################

set new_dataset_path=..\UsefullnessOfDepth\datasets\SynthDet_%dataset_name%
move "%new_dataset_path%\Depth" "%new_dataset_path%\RGB_copy"
move "%new_dataset_path%\Depth_original" "%new_dataset_path%\Depth"
move "%new_dataset_path%\RGB" "%new_dataset_path%\RGB_original"
@REM Copy the directory and its files
xcopy "%new_dataset_path%\Depth" "%new_dataset_path%\RGB" /E /I /Y


@REM Train the model
python utils\train.py ^
--config=configs.SynthDet.SynthDet_%dataset_name%_DFormer_Tiny ^
--gpus 1 ^
--checkpoint_dir %checkpoint_dir% ^
--dataset_type %dataset_type% ^
--x_channels=1 ^
--x_e_channels=1 ^
--num_epochs %num_epochs% ^
--num_hyperparameter_epochs %hyperparam_epochs%


REM Get the last directory starting with "run" from the given path
for /f "delims=" %%d in ('dir /b /ad /on %checkpoint_dir%\SynthDet_%dataset_name%_DFormer-Tiny\run*') do (
    set "last_directory=%%d"
)

REM Get the last filename starting with "epoch" in the last directory
for /f "delims=" %%f in ('dir /b /a-d /on %checkpoint_dir%\SynthDet_%dataset_name%_DFormer-Tiny\%last_directory%\epoch*') do (
    set "last_filename=%%f"
)

echo Last directory: %last_directory%
echo Last filename: %last_filename%

set depth_model_weights=%checkpoint_dir%\SynthDet_%dataset_name%_DFormer-Tiny\%last_directory%\%last_filename%

python utils\evaluate_models.py --config=configs.SynthDet.SynthDet_%dataset_name%_DFormer_Tiny --model_weights %depth_model_weights% --bin_size %bin_size%


@REM shutdown /h