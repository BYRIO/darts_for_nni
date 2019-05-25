"""
File: base_dataset.py
Author: VDeamoV
Email: vincent.duan95@outlook.com
Github: https://github.com/VDeamoV
Description:
    Use to load data, can be use to create meta-dataset
"""
from __future__ import print_function
import os

import yaml
import torch
import torch.utils.data as data
import torchvision
from PIL import Image

from . import functional as F

#  TODO maybe we do not need transforms:  <07-04-19, VDeamoV> # 
class BaseCustomDataset(data.Dataset):
    """
    BaseCustomDataset:
        Provide the fundamental method for other dataset
        Basic function is following below:
            1. Configurae Dataset with yml file
            2. Output Dataset Configure
            3. Basic Analysis the Dataset

    Example yaml:
        ```
        # yaml_file
        info:
            name: 'example_name'
            data_type: ['jpeg', 'JPEG']     # choose which type files you want to load
                                            # case sensitive

        directory:
            root_dir: '~/Desktop/train' # must be absolute path

        preprocess:
            transforms:
                transform_list: ['Scale']
                params:
                    image_size: 224
            transforms_target:
                transform_list: []
                params:
                    
        ``` """

    def __init__(self, yml_file_path):
        super(BaseCustomDataset, self).__init__()
        try:
            with open(yml_file_path, 'r') as config_file:
                params = yaml.load(config_file)
        except FileNotFoundError as error:
            print(error)
        finally:
            print("===yml load success ===")

        self.params = params
        self.name = params['info']['name']
        self.data_type = params['info']['data_type']

        self.root_dir = params['directory']['root_dir']

        self.transforms = params['preprocess']['transforms']
        self.transforms_target = params['preprocess']['transforms_target']

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def config_output(self):
        """
        Output the configure information
        """
        print("======info=======")
        print("Dataset Name: ", self.name)
        print("Data_type: ", self.data_type)
        print("====directory====")
        print("root_dir: ", self.root_dir)
        print("===preprocess===")
        print("transforms: ", self.transforms)
        print("transforms_target: ", self.transforms_target)

    def pre_analyse(self, a, b, c):
        """
        Use to pre_analyse the dataset

        Output:
            1. How many classes
            2. How many samples each class
        """
        #  TODO:  <07-04-19, VDeamoV> #
        raise NotImplementedError


class CustomImageDataset(BaseCustomDataset):
    """
    CustomImageDataset:
        Load Image in the custom folder, then pre analysis the stastic of
        DataSet then save to the params.

        Basic function is folllowing below:
            - find_all_items(self, root_path, debug=False)
                extract all the items in the dir with their label


    The Dir construction:
        ```
        .root
        |--train
        |---|--classA
        |---L--classB
        |
        L--test
            |--classA
            L--classB
        ```

    """

    def __init__(self, yml_file_path, mode='train', debug=False):
        """
        inherit from BaseCustomDataset

        inherit PARAMS:
            self.name
            self.data_type
            self.root_dir
            self.transforms
            self.transforms_target

        new PARAMS:
            params: mode <string> train / val / test to choose which dir to load
            params: yml_file_path <string>: your yaml config_path
            params: debug <boolen>: defaut=False
                        if debug set true, it will print the label and data_path
                        in real time

        new custom ouput:(these value is essential for meta_dataset_loader)
            - num_train_classes
            - num_val_classes
            - num_test_classes
            - num_per_train_classes
            - num_per_val_classes
            - num_per_test_classes

            !!!We suppose each class have the same number of samples!!!
            !!!We suppose we DO NOT have HIDDEN DIR in the directory!!!
        """
        # create data here
        super(CustomImageDataset, self).__init__(yml_file_path)
        #  TODO: put find num_train_classes eg outof the if# 

        if mode == 'train':
            path = os.path.join(self.root_dir, 'train')

            # list the dir in the path
            dir_list = F.list_dir(path)
            self.num_train_classes = len(dir_list)

            # output how many files in the class
            files = os.listdir(os.path.join(path, dir_list[0]))
            self.num_per_train_classes = len(files)

            self.out_x, self.out_y = F.find_all_items(self.data_type, path, debug=debug)

            # change label from string to index
            self.out_y, _ = F.convert_strlist_intlist(self.out_y)


        elif mode == 'val':
            path = os.path.join(self.root_dir, 'val')

            dir_list = F.list_dir(path)
            self.num_val_classes = len(dir_list)

            files = os.listdir(os.path.join(path, dir_list[0]))
            self.num_per_val_classes = len(files)

            self.out_x, self.out_y = F.find_all_items(self.data_type, path, debug=debug)

            # change label from string to index
            self.out_y, _ = F.convert_strlist_intlist(self.out_y)

        elif mode == 'test':
            path = os.path.join(self.root_dir, 'test')

            dir_list = F.list_dir(path)
            self.num_test_classes = len(dir_list)

            files = os.listdir(os.path.join(path, dir_list[0]))
            self.num_per_test_classes = len(files)

            self.out_x, self.out_y = F.find_all_items(self.data_type, path, debug=debug)

            # change label from string to index
            self.out_y, _ = F.convert_strlist_intlist(self.out_y)
        else:
            raise ValueError

    def __getitem__(self, index):
        # output a batch
        # we don't need to concern about the batch size
        # index is one number, so ouput is one image
        #  TODO:  add transform label to number#

        # imply transforms
        raw_image = Image.open(self.out_x[index])
        if raw_image.getbands()[0] == 'L':
            raw_image = raw_image.convert('RGB')

        if self.transforms['transform_list']:
            transform_params = self.transforms['params']
            transform_list = [F.parse_transform(types, transform_params) \
                              for types in self.transforms['transform_list']]
            trans = torchvision.transforms.Compose(transform_list)
            processed_image = trans(raw_image)
        else:
            transform_to_tensor = [torchvision.transforms.ToTensor()]
            trans = torchvision.transforms.Compose(transform_to_tensor)
            processed_image = trans(raw_image)

        # impy target tansforms
        if self.transforms_target['transform_list']:
            transform_params = self.transforms_target['params']
            transform_list = [F.parse_transform(types, transform_params) \
                              for types in self.transforms_target['transform_list']]
            trans_target = torchvision.transforms.Compose(transform_list)
            processed_label = trans_target(self.out_y[index])
        else:
            processed_label = self.out_y[index]

        return processed_image, processed_label

    def __len__(self):
        return len(self.out_y)


def test():
    """
    some test for method
    """
    dataset = CustomImageDataset("./test.yml", mode='train', debug=debug)
    # test for out_y, out_x
    print(dataset.out_x)
    print(dataset.out_y)
    input("press to continue")
    test_image = dataset.__getitem__(0)
    # load test
    if not test_image:
        raise Exception("DID NOT LOAD IN DATA, please check for ROOT_PATH")
    print("===== LOADTEST SUCCESS =====")
    # preprocess test
    if not isinstance(test_image[0], torch.Tensor):
        raise Exception("transform_list DIDN'T WORK")
    print("===== PREPROCESS SUCCESS =====")
    print(dataset.num_train_classes)
    print("===== Test for num_train_classes COMPELETE =====")


if __name__ == "__main__":
    test()
