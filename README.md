# world_model
This repository is used to store the code for my master's thesis.

Experiments here: https://tks-zx.fzi.de/projects/e7fe0f4551b74ce79bed2431503110ec/experiments?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&order=-last_update&filter=

## training
run
```angular2html
python train.py --conifg-file mile/configs/your_config.yml
```
You can use default config file 'mile/configs/mile.yml', or create your own config file in 'mile/configs/'.\
In config file(*.yml), you can set all the configs listed in 'mile/config.py'.\
Before training, make sure that the required input data/output reconstruction data as well as the model structure/dimensions are correctly set in 'mile/configs/your_config.yml'.

## test
run
```angular2html
python prediction.py --config-file mile/configs/prediction.yml
```
The config file is the same as in training.\
In file 'mile/data/dataset.py', class 'DataModule', function 'setup', you can change the predict_dataset/sampler type.