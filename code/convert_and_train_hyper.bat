REM This script is used to convert the dataset from SOLO format to COCO format and then to DFormer format
set dataset_location=%1
set dataset_name=%2
set dataset_type=%3

set use_edge_enhancement=false

set checkpoint_dir=checkpoints

IF %dataset_type% == "" (
    set dataset_type=groceries
)

IF EXIST "DFormer\datasets\SynthDet_%dataset_name%" goto :skip_dataset_creation

@REM Creating dataset
solo2coco %dataset_location% data\SynthDet\

cd SynthDet_Pascal
python convert_solo_depth_to_coco.py %dataset_location% ..\data\SynthDet\coco\depth

@REM Rename the dataset to the correct name
move "..\data\SynthDet\coco" "..\data\SynthDet\coco_%dataset_name%"

set dataset_path=..\data\SynthDet\coco_%dataset_name%

python convert_coco_to_dformer.py %dataset_path% ..\DFormer\datasets\SynthDet_%dataset_name% %dataset_type%

IF %use_edge_enhancement% == true (
    python add_edge_enhancement.py ..\DFormer\datasets\SynthDet_%dataset_name%
)

@REM Remove the dataset from the data folder
rmdir /s /q %dataset_path%

@REM Change to the DFormer directory
cd ..\DFormer

:skip_dataset_creation
cd DFormer

@REM Copy the template file to the new file
copy local_configs\SynthDet\SynthDet_template_DFormer_Tiny.py local_configs\SynthDet\SynthDet_%dataset_name%_DFormer_Tiny.py
cd local_configs\SynthDet

@REM Change back to DFormer
cd ..\..

@REM Train the model
python utils\train_clean.py --config=local_configs.SynthDet.SynthDet_%dataset_name%_DFormer_Tiny --gpus 1 --checkpoint_dir %checkpoint_dir% --dataset_type %dataset_type% --x_channels=3 --x_e_channels=1

@REM Create prediction mious for the dataset with RGB-Depth

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

python utils\evaluate_models.py --config=local_configs.SynthDet.SynthDet_%dataset_name%_DFormer_Tiny --model_weights %rgb_depth_model_weights%

set new_dataset_path=..\DFormer\datasets\SynthDet_%dataset_name%
move "%new_dataset_path%\Depth" "%new_dataset_path%\Depth_original"
xcopy "%new_dataset_path%\RGB" "%new_dataset_path%\Depth" /E /I /Y


@REM Train the model
python utils\train_clean.py --config=local_configs.SynthDet.SynthDet_%dataset_name%_DFormer_Tiny --gpus 1 --checkpoint_dir %checkpoint_dir% --dataset_type %dataset_type% --x_channels=3 --x_e_channels=3


@REM Create prediction mious for the dataset with RGB-Black

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

python utils\evaluate_models.py --config=local_configs.SynthDet.SynthDet_%dataset_name%_DFormer_Tiny --model_weights %rgb_only_model_weights%


@REM Evaluate the models
@REM python utils\create_predictions.py --model_a_path %rgb_only_model_weights% --model_b_path %rgb_depth_model_weights% --dir_dataset datasets\SynthDet_%dataset_name% --config=local_configs.SynthDet.SynthDet_%dataset_name%_DFormer_Tiny --test_mode
python utils\create_predictions.py --model_a_path %rgb_only_model_weights% --model_b_path %rgb_depth_model_weights% --dir_dataset datasets\SynthDet_%dataset_name% --config=local_configs.SynthDet.SynthDet_%dataset_name%_DFormer_Tiny

@REM Create gen_results.txt and store the last command
echo utils\create_predictions.py --model_a_path %rgb_only_model_weights% --model_b_path %rgb_depth_model_weights% --dir_dataset datasets\SynthDet_%dataset_name% --config=local_configs.SynthDet.SynthDet_%dataset_name%_DFormer_Tiny >> %checkpoint_dir%\SynthDet_%dataset_name%_DFormer-Tiny\gen_results.txt

@REM ####################################### DEPTH ONLY #######################################

@REM set new_dataset_path=..\DFormer\datasets\SynthDet_%dataset_name%
@REM @REM move "%new_dataset_path%\Depth" "%new_dataset_path%\Depth_original"
@REM @REM move "%new_dataset_path%\Grayscale" "%new_dataset_path%\Depth"
@REM move "%new_dataset_path%\Depth" "%new_dataset_path%\RGB_copy"
@REM move "%new_dataset_path%\Depth_original" "%new_dataset_path%\Depth"
@REM move "%new_dataset_path%\RGB" "%new_dataset_path%\RGB_original"
@REM @REM Copy the directory and its files
@REM xcopy "%new_dataset_path%\Depth" "%new_dataset_path%\RGB" /E /I /Y


@REM @REM Train the model
@REM python utils\train_clean.py --config=local_configs.SynthDet.SynthDet_%dataset_name%_DFormer_Tiny --gpus 1 --checkpoint_dir %checkpoint_dir% --dataset_type %dataset_type% --x_channels=1 --x_e_channels=1


@REM @REM Create prediction mious for the dataset with RGB-Black

@REM REM Get the last directory starting with "run" from the given path
@REM for /f "delims=" %%d in ('dir /b /ad /on %checkpoint_dir%\SynthDet_%dataset_name%_DFormer-Tiny\run*') do (
@REM     set "last_directory=%%d"
@REM )

@REM REM Get the last filename starting with "epoch" in the last directory
@REM for /f "delims=" %%f in ('dir /b /a-d /on %checkpoint_dir%\SynthDet_%dataset_name%_DFormer-Tiny\%last_directory%\epoch*') do (
@REM     set "last_filename=%%f"
@REM )

@REM echo Last directory: %last_directory%
@REM echo Last filename: %last_filename%

@REM set depth_model_weights=%checkpoint_dir%\SynthDet_%dataset_name%_DFormer-Tiny\%last_directory%\%last_filename%

@REM python utils\evaluate_models.py --config=local_configs.SynthDet.SynthDet_%dataset_name%_DFormer_Tiny --model_weights %depth_model_weights%
