# MscPascalBenschop: Why is Depth useful for Computer Vision?

Repository for the master thesis project of Pascal Benschop. 

## SynthDet

### Setup

clone SynthDet from unity: https://github.com/Unity-Technologies/SynthDet

Run simulations via Unity, enable depth labeler.

Activate conda environment:

Step 1
```
conda env create -f environment.yaml

conda env update --file environment2.yml
```
If an error occurs with something like: `An error occurred while installing package 'defaults::xz-5.4.5-h8cc25b3_0'.
Rolling back transaction: done`

Then do `conda env remove --name python_environment` and `conda env create -f environment.yaml`

Step 2
```
conda activate python_environment

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html

pip install datasetinsights
```

For Grad-CAM

Navigate to pytorch-grad-cam, and once inside:
```
pip install .
```

Navigate to Synthdet_Pascal

Convert unity dataset in SOLO format to COCO format:
```
solo2coco path\to\UnityTechnologies\SynthDet\solo_x ..\data\SynthDet\
```

Add depth to COCO dataset
```
python convert_solo_depth_to_coco.py path\to\UnityTechnologies\SynthDet\solo_x ..\data\SynthDet\coco\depth
```


## DFormer

Inside SynthDet_Pascal
```
python convert_coco_to_dformer.py ..\data\SynthDet\coco\ ..\DFormer\datasets\SynthDet_{new_name}
```

### Training

Navigate to DFormer repository
```
python .\utils\train.py --config=local_configs.SynthDet.SynthDet_{dataset_name}_DFormer_Tiny --gpus 1
```



## Simple Segmentation network
Inside src/SynthDet_Segmentation

### Training
In the config can decide what to use
TODO change channels to config
```
python train.py --config=config.SynthDet3_Segmodel --channels=4
```
Look at tensorboard:
```
tensorboard --logdir=checkpoints --host localhost --port 8888
```

## Plotting

See documentation [here](code/DFormer_Pascal/README.md)

## Deeplab-xception
Inside code/

### Training

```
python .\pytorch-deeplab-xception\train.py --config DFormer.local_configs.SUNRGBD.SUNRGBD_DFormer_Tiny --resume .\pytorch-deeplab-xception\checkpoints\run\SUNRGBD\deeplab-xception\model_best.pth.tar --lr 0.001 --ft
```


## Sources

[SynthDet](https://github.com/Unity-Technologies/SynthDet)
[2.5 Malleable Convolution](https://github.com/charlesCXK/RGBD_Semantic_Segmentation_PyTorch)


Apparently bincount is very slow with a lot of the same class https://discuss.pytorch.org/t/torch-bincount-1000x-slower-on-cuda/42654/2 






<!-- ## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers. -->
