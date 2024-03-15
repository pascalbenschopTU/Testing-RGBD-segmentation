REM This script is used to convert the dataset from SOLO format to COCO format and then to DFormer format
set dataset_location=%1
set dataset_name=%2
set use_gems=%3

set use_edge_enhancement=true

IF "%use_gems%"=="" (
    set use_gems=false
)

IF EXIST "DFormer\datasets\SynthDet_%dataset_name%" goto :skip_dataset_creation

@REM Creating dataset
solo2coco %dataset_location% data\SynthDet\

cd SynthDet_Pascal
python convert_solo_depth_to_coco.py %dataset_location% ..\data\SynthDet\coco\depth

@REM Rename the dataset to the correct name
move "..\data\SynthDet\coco" "..\data\SynthDet\coco_%dataset_name%"

set dataset_path=..\data\SynthDet\coco_%dataset_name%

IF %use_gems% == true (
    @REM Convert the dataset to DFormer format
    python convert_coco_to_dformer.py %dataset_path% ..\DFormer\datasets\SynthDet_%dataset_name% --gems
) ELSE (
    @REM Convert the dataset to DFormer format
    python convert_coco_to_dformer.py %dataset_path% ..\DFormer\datasets\SynthDet_%dataset_name%
)

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
copy local_configs\SynthDet\SynthDet_template_DFormer_Tiny.py local_configs\SynthDet\SynthDet_%dataset_name%_Dformer_Tiny.py
cd local_configs\SynthDet

IF %use_gems% == true (
    @REM Use sed to replace the dataset name and classes in the new file
    powershell -Command "(gc SynthDet_%dataset_name%_Dformer_Tiny.py) -replace 'C.dataset_name = .*', \"C.dataset_name = 'SynthDet_%dataset_name%'\" -replace 'classes = \"groceries\"', 'classes = \"gems\"' | Out-File -encoding ASCII SynthDet_%dataset_name%_Dformer_Tiny.py"
) ELSE (
    @REM Use sed to replace the dataset name in the new file
    powershell -Command "(gc SynthDet_%dataset_name%_Dformer_Tiny.py) -replace 'C.dataset_name = .*', \"C.dataset_name = 'SynthDet_%dataset_name%'\" | Out-File -encoding ASCII SynthDet_%dataset_name%_Dformer_Tiny.py"
)

@REM Change back to DFormer
cd ..\..

@REM Train the model
python utils\train.py --config=local_configs.SynthDet.SynthDet_%dataset_name%_Dformer_Tiny --gpus 1

@REM Create prediction mious for the dataset with RGB-Depth

REM Get the last directory starting with "run" from the given path
for /f "delims=" %%d in ('dir /b /ad /on checkpoints\SynthDet_%dataset_name%_DFormer-Tiny\run*') do (
    set "last_directory=%%d"
)

REM Get the last filename starting with "epoch" in the last directory
for /f "delims=" %%f in ('dir /b /a-d /on checkpoints\SynthDet_%dataset_name%_DFormer-Tiny\%last_directory%\epoch*') do (
    set "last_filename=%%f"
)

echo Last directory: %last_directory%
echo Last filename: %last_filename%

set rgb_depth_model_weights=checkpoints\SynthDet_%dataset_name%_DFormer-Tiny\%last_directory%\%last_filename%

python utils\evaluate_models.py --config=local_configs.SynthDet.SynthDet_%dataset_name%_Dformer_Tiny --model_weights %rgb_depth_model_weights%

set new_dataset_path=..\DFormer\datasets\SynthDet_%dataset_name%
@REM Change the depth of the dataset to black
move "%new_dataset_path%\Depth" "%new_dataset_path%\Depth_original"
move "%new_dataset_path%\Grayscale" "%new_dataset_path%\Depth"

@REM Train the model
python utils\train.py --config=local_configs.SynthDet.SynthDet_%dataset_name%_Dformer_Tiny --gpus 1


@REM Create prediction mious for the dataset with RGB-Black

REM Get the last directory starting with "run" from the given path
for /f "delims=" %%d in ('dir /b /ad /on checkpoints\SynthDet_%dataset_name%_DFormer-Tiny\run*') do (
    set "last_directory=%%d"
)

REM Get the last filename starting with "epoch" in the last directory
for /f "delims=" %%f in ('dir /b /a-d /on checkpoints\SynthDet_%dataset_name%_DFormer-Tiny\%last_directory%\epoch*') do (
    set "last_filename=%%f"
)

echo Last directory: %last_directory%
echo Last filename: %last_filename%

set rgb_black_model_weights=checkpoints\SynthDet_%dataset_name%_DFormer-Tiny\%last_directory%\%last_filename%

python utils\evaluate_models.py --config=local_configs.SynthDet.SynthDet_%dataset_name%_Dformer_Tiny --model_weights %rgb_black_model_weights%


@REM Evaluate the models
python utils\create_predictions.py --model_a_path %rgb_black_model_weights% --model_b_path %rgb_depth_model_weights% --dir_dataset datasets\SynthDet_%dataset_name% --config=local_configs.SynthDet.SynthDet_%dataset_name%_Dformer_Tiny

@REM Create gen_results.txt and store the last command
echo utils\create_predictions.py --model_a_path %rgb_black_model_weights% --model_b_path %rgb_depth_model_weights% --dir_dataset datasets\SynthDet_%dataset_name% --config=local_configs.SynthDet.SynthDet_%dataset_name%_Dformer_Tiny > checkpoints\SynthDet_%dataset_name%_DFormer-Tiny\gen_results.txt
