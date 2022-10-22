# CvT-for-MTHV2

# Requirement
  torch 1.7.1

# Dataset
  MTHV2 train:test = 4:1  
  :link:Download processed dataset:(https://1drv.ms/u/s!Aj6X7kgt6NgZjRhnhV_dKIRZLYL5?e=ZUfgGI)
  The dataset used for calculating AR and CR metrics is also included.
  
# Code
  Configurations have been revised for MTHV2
  
# Main results
## Models trained on MTHV2
| Model  | Resolution | Top-1    | Top-5  | Recall |  F1   |  CR  |  AR  |1-N.E.D|
|--------|------------|-------   |--------|------- |-------|------|------|-------|
| CvT-13 | 224x224    | 97.27%   | 98.91% | 97.27% |97.13% |90.13%|90.07%|90.09% |

:link:You can download all the models from (https://drive.google.com/drive/folders/1JlxLm0VVYAQCdgx-rcQMWMxfobPHrTXw?usp=sharing).  

models should be placed at 

``` sh
CvT-for-MTHV2/OUTPUT/mthv2/cvt-13-224x224/
```

### Training on local machine

``` sh
bash run.sh -g 8 -t train --cfg experiments/imagenet/cvt/cvt-13-224x224.yaml
```

You can also modify the config parameters from the command line. For example, if you want to change the lr rate to 0.1, you can run the command:
``` sh
bash run.sh -g 8 -t train --cfg experiments/imagenet/cvt/cvt-13-224x224.yaml TRAIN.LR 0.1
```

Notes:
- The checkpoint, model, and log files will be saved in OUTPUT/{dataset}/{training config} by default.

### Testing pre-trained models
``` sh
bash run.sh -t test --cfg experiments/mthv2/cvt/cvt-13-224x224.yaml TEST.MODEL_FILE OUTPUT/mthv2/cvt-13-224x224/model_best.pth
```
  
### Calculating Editing Distance
``` sh
bash run.sh -t test_ED --cfg experiments/mthv2/cvt/cvt-13-224x224.yaml TEST.MODEL_FILE OUTPUT/mthv2/cvt-13-224x224/model_best.pth
```
  
## New function
* precision, recall and F1 score :white_check_mark:
* CR, AR :white_check_mark:

:cherries:More deltials will be added soon!
