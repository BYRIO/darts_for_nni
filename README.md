Darts for NNI
=============
<!-- vim-markdown-toc GitLab -->

* [Advantages of DARTS](#advantages-of-darts)
* [Using Manual](#using-manual)
* [Config Tuner YAML](#config-tuner-yaml)
  * [Config Model Search Dataset](#config-model-search-dataset)
  * [Config Primitive for Ops](#config-primitive-for-ops)
    * [Example YAML](#example-yaml)
  * [How to Custom Ops](#how-to-custom-ops)
* [File Structure Manual](#file-structure-manual)
    * [OutputFile Structure](#outputfile-structure)
* [Reference](#reference)
* [FutureWork](#futurework)

<!-- vim-markdown-toc -->
Advantages of DARTS
-----
 
DARTS (Differentiable ARchiTecture Search) is an extremely efficient model searching algorithm for convolutional neural networks and recurent neural networks (not implemented).  
Instead of searching over a discrete set of candidate architectures,  the algorithm relax the search space to be continuous, so that the architecture can be optimized with respect to its validation set performance by gradient descent. This feature brings a great advantage of tremendous searching efficiency and training speed, which outperforms other related algorithms by three orders of magnitudes.

*ps. Final file constructure and paper link will be found in the following part of this description.*

Using Manual
------------
## Config Tuner YAML
### Config Model Search Dataset
```bash
# Config Dataset Name in the tuner.yaml
classArgs:
  dataset_name: # CIFAR or CUSTOM
  dataset_path: /path/to/the/cifar/folder  # only need for cifar
  custom_yaml: /path/to/the/custom/dataset_yaml # only need when choose CUSTOM

# in the dataset_yaml
```

### Config Primitive for Ops
```bash
# After you config your ops in the `operations.py`, you can use it here
classArgs:
  primitives: ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'your_ops']
```

#### Example YAML
>  TODO: need fix for integrate into the darts > 
```bash
authorName: Occam
experimentName: test_darts
trialConcurrency: 1 # we only need 1 trial we don't run more than one 
maxExecDuration: 1000h
maxTrialNum: 1
trainingServicePlatform: local
useAnnotation: False
tuner:
    codeDir: /home/apex/DeamoV/github/darts_for_nni
    classFileName: custom_tuner.py
    className: custom_tuner
    # Any parameter need to pass to your tuner class __init__ constructor
    # can be specified in this optional classArgs field, for example 
    classArgs:
        model_architecture_path: "path"
        primitives: ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']
        dataset_path: "/home/apex/DeamoV/github/darts_for_nni/darts_source/data"
        dataset_name: "CUSTOM" # CIFAR & CUSTOM
        custom_yaml: "/home/apex/DeamoV/github/darts_for_nni/custom_mini_imagenet.yaml" # only need when dataset_name is CUSTOM
        output_path: "/home/apex/tmp/testoutput"
        # this is to output all the log and script
trial:
    command: "python3 train_search.py"
    codeDir: "/home/apex/DeamoV/github/darts_for_nni/darts_source"
    gpuNum: 2
    #  TODO: add multi gpu later # 
```

### How to Custom Ops
```bash
# file: $DARTS_FOR_NNI_ROOT_PATH/darts_source/operations.py

# Add operations to OPS
OPS = {
    'none':\
        lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3':\
        lambda C, stride, affine: nn.AvgPool2d(3, stride=stride,
                                               padding=1, count_include_pad=False),
    'max_pool_3x3':\
        lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect':\
        lambda C, stride, affine:\
            Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3':\
        lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5':\
        lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7':\
        lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3':\
        lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5':\
        lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7':\
        lambda C, stride, affine: nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
            nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(C, affine=affine)
        ),
    'your_choose':\
        lambda C, stride, affine: your_ops
}
# above is the default options in OPS

```
File Structure Manual
--------------------
#### OutputFile Structure
1. the final structure of model searching (cell)
2. the image of final structure (.png)
3. model params

```bash
.
├── log.txt # trained log
├── normal_epoch_0_valid_acc_0.0_train_acc_39.484375.png # normal graph pic
├── reduce_epoch_0_valid_acc_0.0_train_acc_39.484375.png # reduce graph pic
├── scripts
│   ├── genotypes.py
│   ├── model.py
│   ├── operations.py
│   ├── test_imagenet.py                                 # test script on imagenet
│   ├── train_imagenet.py                                # train script on imagenet
│   └── utils.py
└── weights.pt                                           # trained weights
```

Reference
---------
- [DARTS paper](https://arxiv.org/pdf/1806.09055)
- [DARTS_official_code](https://github.com/quark0/darts)

FutureWork
----------
- [ ] Add support for multi-GPUs
- [ ] Redefine the output axis' name
- [ ] Custom model_search's params
