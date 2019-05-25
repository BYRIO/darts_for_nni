"""
File: functional.py
Author: VDeamoV
Email: vincent.duan95@outlook.com
Github: https://github.com/VDeamoV
Description:
    This file is used to provide the basic function which used in dataset
    eg.
    ```python
    import functional as F
    K.parse_transform
    ```
"""

import os
import torchvision.transforms as transforms

def list_dir(dir_path, debug=False):
    """
    List the dir in the current path
    """
    _tmp = os.listdir(dir_path)
    dir_list = [x for x in _tmp if os.path.isdir(os.path.join(dir_path, x))]
    return dir_list

def convert_strlist_intlist(str_labels):
    """
    This fucntion is used to change str_type labels
    to index_labels

    params: string type list labels <list> 

    int_labels: int type list labels <list>
    search_index: string type unique lables <list>

    ```python
    >>> a = ["a", "b", "c", "c"]
    >>> int_labels, search_index = convert_strlist_intlist(a)
    >>> int_labels
    >>> [0, 1, 2, 2]
    >>> search_index
    >>> ["a", "b", "c"]
    ```
    """
    search_index = []
    int_labels = []
    for _ in str_labels:
        if _ in search_index:
            int_labels.append(search_index.index(_))
        else:
            search_index.append(_)
            int_labels.append(len(search_index)-1)

    return int_labels, search_index

def find_all_items(data_type, root_path, debug=False):
    """
    extract all the items in the dir with their label

    params: root_path <str>

    output: output_x <list> date_paths
    iutput: output_y <list> data's label
    """
    output_x = []
    output_y = []
    _walk_instance = os.walk(root_path)
    for root, _, files in _walk_instance:
        for item in files:
            if item.endswith(tuple(data_type)):
                if debug:
                    print("image_path:", os.path.join(root_path, root.split('/')[-1], item))
                    print("label:", root.split('/')[-1])
                raw_image_path = os.path.join(root_path, root.split('/')[-1], item)
                output_x.append(raw_image_path)
                output_y.append(root.split('/')[-1])
    return output_x, output_y

def parse_transform(transform_type, transform_params):
    """
    For quick apply image preprocess by using torchvision.transforms
    """
    #  TODO: Support more method in it# 
    try:
        method = getattr(transforms, transform_type)
    except:
        raise Exception('Do not find the transform type in torchvision.transforms.\
                        Please double check!')

    if transform_type in ('RandomSizedCrop', 'CenterCrop'):
        if transform_params['image_size']:
            return method(transform_params['image_size'])
        raise Exception('Please configure image_size in your config file')

    if transform_type == 'Scale':
        raise Exception('Please use Resize, it is same as Scale')

    if transform_type == 'Resize':
        if transform_params['resize']:
            size = transform_params['resize']
            return method(size)
        raise Exception('Please configure scale in your config file')

    if transform_type == 'Normalize':
        if transform_params['mean'] and transform_params['std']:
            return method(mean=transform_params['mean'], std=transform_params['std'])
        raise Exception('Please configure mean and std in your config file')

    if transform_type == 'ColorJitter':
        if transform_params['colorjitter_params']:
            jitter_params = transform_params['colorjitter_params']
            # 0 : hue
            # 1 : saturation
            # 2 : contrast
            # 3 : brightness
            return method(hue=jitter_params[0],
                          saturation=jitter_params[1],
                          contrast=jitter_params[2],
                          brightness=jitter_params[3])
        raise Exception('Please configure your params in your config file')

    transforms.ColorJitter
    if transform_type == 'RandomHorizontalFlip':
        # default prob is 0.5, you don't need to change it
        return method()

    if transform_type == 'ToTensor':
        return method()

    else:
        raise Exception('Unsupport Transform Type')

