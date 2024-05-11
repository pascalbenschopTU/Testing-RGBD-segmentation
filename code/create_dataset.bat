REM This script is used to convert the dataset from SOLO format to COCO format and then to DFormer format
set dataset_location=%1
set dataset_name=%2
set dataset_type=%3
set test_only=%4

@REM set min_depth=50
set min_depth=90
set max_depth=110

set use_edge_enhancement=false

set checkpoint_dir=checkpoints

IF "%dataset_type%" == "" (
    set dataset_type=groceries
)

IF "%test_only%" == "" (
    set test_only=False
)

@REM IF EXIST "DFormer\datasets\SynthDet_%dataset_name%" goto :skip_dataset_creation

@REM Creating dataset
solo2coco %dataset_location% data\SynthDet\

cd UsefullnessOfDepth\SynthDet_tools
python convert_solo_depth_to_coco.py %dataset_location% ..\..\data\SynthDet\coco\depth --min_depth %min_depth% --max_depth %max_depth%

@REM Rename the dataset to the correct name
move "..\..\data\SynthDet\coco" "..\..\data\SynthDet\coco_%dataset_name%"

set dataset_path=..\..\data\SynthDet\coco_%dataset_name%
@REM code\UsefullnessOfDepth
python convert_coco_to_dformer.py %dataset_path% ..\datasets\SynthDet_%dataset_name% %dataset_type% --test_mode %test_only%

IF %use_edge_enhancement% == true (
    python add_edge_enhancement.py ..\datasets\SynthDet_%dataset_name%
)

@REM Remove the dataset from the data folder
rmdir /s /q %dataset_path%

@REM Change to the DFormer directory
cd ..\..