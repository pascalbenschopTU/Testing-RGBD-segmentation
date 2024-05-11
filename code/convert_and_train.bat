REM This script is used to convert the dataset from SOLO format to COCO format and then to UsefullnessOfDepth format
set dataset_location=%1
set dataset_name=%2
set dataset_type=%3
set num_epochs=%4
set hyperparam_epochs=%5
@REM Choose model from [DFormer-Tiny, DFormer-Large, CMX-B2, DeepLab, segformer]
set model=%6

set use_edge_enhancement=false
set checkpoint_dir=checkpoints
@REM set checkpoint_dir=checkpoints_CMX
set bin_size=1

IF "%num_epochs%" == "" (
    set num_epochs=60
)

IF "%hyperparam_epochs%" == "" (
    set hyperparam_epochs=5
)

IF "%model%" == "" (
    set model=DFormer-Tiny
)

IF "%dataset_name%" == "SUNRGBD" (
    set config=configs.SUNRGBD.SUNRGBD_DFormer_Tiny
    goto :skip_config_creation
)

for /f "tokens=1,2 delims=-" %%a in ("%model%") do (
    set model_name=%%a
    set model_type=%%b
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
copy configs\SynthDet\SynthDet_template_DFormer_Tiny.py configs\SynthDet\SynthDet_%dataset_name%_%model_name%.py
set config=configs.SynthDet.SynthDet_%dataset_name%_%model_name%

:skip_config_creation
cd UsefullnessOfDepth

REM config = %config%

@REM ####################################### RGB Depth #######################################

@REM Train the model
python utils\train.py ^
--config=%config% ^
--gpus 1 ^
--checkpoint_dir %checkpoint_dir% ^
--dataset_type %dataset_type% ^
--x_channels=3 ^
--x_e_channels=1 ^
--num_epochs %num_epochs% ^
--num_hyperparameter_epochs %hyperparam_epochs% ^
--model %model%

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

python utils\evaluate_models.py --config=%config% --model_weights %rgb_depth_model_weights% --bin_size %bin_size% --model %model%

@REM if model is DeepLab or DFormer_Large, skip the RGB only part
IF "%model%" == "DFormer_Large" goto end
IF "%model%" == "DeepLab" goto end
IF "%model%" == "CMX_B2" goto end
IF "%model%" == "segformer" goto end

@REM ####################################### RGB ONLY #######################################

@REM @REM Train the model
@REM python utils\train.py ^
@REM --config=%config% ^
@REM --gpus 1 ^
@REM --checkpoint_dir %checkpoint_dir% ^
@REM --dataset_type %dataset_type% ^
@REM --x_channels=3 ^
@REM --x_e_channels=3 ^
@REM --num_epochs %num_epochs% ^
@REM --num_hyperparameter_epochs %hyperparam_epochs% ^
@REM --model %model%

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

@REM set rgb_only_model_weights=%checkpoint_dir%\SynthDet_%dataset_name%_DFormer-Tiny\%last_directory%\%last_filename%

@REM python utils\evaluate_models.py --config=%config% --model_weights %rgb_only_model_weights% --bin_size %bin_size% --model %model%


@REM @REM Create gen_results.txt and store the last command
@REM echo utils\create_predictions.py --model_a_path %rgb_only_model_weights% --model_b_path %rgb_depth_model_weights% --dir_dataset datasets\SynthDet_%dataset_name% --config=configs.SynthDet.SynthDet_%dataset_name%_DFormer_Tiny >> %checkpoint_dir%\SynthDet_%dataset_name%_DFormer-Tiny\gen_results.txt


@REM ####################################### DEPTH ONLY #######################################

@REM @REM Train the model
@REM python utils\train.py ^
@REM --config=%config% ^
@REM --gpus 1 ^
@REM --checkpoint_dir %checkpoint_dir% ^
@REM --dataset_type %dataset_type% ^
@REM --x_channels=1 ^
@REM --x_e_channels=1 ^
@REM --num_epochs %num_epochs% ^
@REM --num_hyperparameter_epochs %hyperparam_epochs% ^
@REM --model %model%


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

@REM python utils\evaluate_models.py --config=%config% --model_weights %depth_model_weights% --bin_size %bin_size% --model %model%

:end
@REM shutdown /h