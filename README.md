# MUVO
This is PyTorch implementation for 
> MUVO: A Multimodal Generative World Model for Autonomous Driving with Geometric Representations. <br/>
> [arXiv](https://arxiv.org/abs/2311.11762)


## Requirements
The simplest way to install all required dependencies is to create 
a [conda](https://docs.conda.io/projects/miniconda/en/latest/) enviroment by running
```
conda env create -f conda_env.yml
```
Then activate conda environment by
```
conda activate muvo
```
or create your own venv and install the requirement by running
```
pip install -r requirements.txt
```


## Dataset
Use [CARLA](http://carla.org/) to collection data. 
First install carla refer to its [documentation](https://carla.readthedocs.io/en/latest/).

### Dataset Collection
Change settings in config/, 
then run `bash run/data_collect.sh ${PORT}` 
with `${PORT}` the port to run CARLA (usually `2000`) <br/>
The data collection code is modified from 
[CARLA-Roach](https://github.com/zhejz/carla-roach) and [MILE](https://github.com/wayveai/mile),
some config settings can be referred there.

### Voxelization
After collecting the data by CARLA, create voxels data by running `data/generate_voxels.py`, <br/> 
voxel settings can be changed in `data_preprocess.yaml`.

### Folder Structure
After completing the above steps, or otherwise obtaining the dataset,
please change the file structure of the dataset. <br/>

The data is organized in the following format
```
/carla_dataset/trainval/
                   ├── train/
                   │     ├── Town01/
                   │     │     ├── 0000/
                   │     │     │     ├── birdview/
                   │     │     │     │      ├ birdview_000000000.png
                   │     │     │     │      .
                   │     │     │     ├── depth_semantic/
                   │     │     │     │      ├ depth_semantic_000000000.png
                   │     │     │     │      .
                   │     │     │     ├── image/
                   │     │     │     │      ├ image_000000000.png
                   │     │     │     │      .
                   │     │     │     ├── points/
                   │     │     │     │      ├ points_000000000.png
                   │     │     │     │      .
                   │     │     │     ├── points_semantic/
                   │     │     │     │      ├ points_semantic_000000000.png
                   │     │     │     │      .
                   │     │     │     ├── routemap/
                   │     │     │     │      ├ routemap_000000000.png
                   │     │     │     │      .
                   │     │     │     ├── voxel/
                   │     │     │     │      ├ voxel_000000000.png
                   │     │     │     │      .
                   │     │     │     └── pd_dataframe.pkl
                   │     │     ├── 0001/
                   │     │     ├── 0002/
                   │     |     .
                   │     |     └── 0024/
                   │     ├── Town03/
                   │     ├── Town04/
                   │     .
                   │     └── Town06/
                   ├── val0/
                   .
                   └── val1/
```

## training
Run
```angular2html
python train.py --conifg-file muvo/configs/your_config.yml
```
You can use default config file `muvo/configs/muvo.yml`, or create your own config file in `muvo/configs/`. <br/>
In `config file(*.yml)`, you can set all the configs listed in `muvo/config.py`. <br/>
Before training, make sure that the required input/output data as well as the model structure/dimensions are correctly set in `muvo/configs/your_config.yml`.

## test
Run
```angular2html
python prediction.py --config-file muvo/configs/prediction.yml
```
The config file is the same as in training.\
In `file 'muvo/data/dataset.py', class 'DataModule', function 'setup'`, you can change the test dataset/sampler type.

## Related Projects
Our code is based on [MILE](https://github.com/wayveai/mile). 
And thanks to [CARLA-Roach](https://github.com/zhejz/carla-roach) for making a gym wrapper around CARLA.