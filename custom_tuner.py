"""
File: custom_tuner.py
Author: OccamRazer
Email: vincent.duan95@outlook.com
Github: https://github.com/VDeamoV
Description:
    Change Name for futher use
"""
from nni.tuner import Tuner


class custom_tuner(Tuner):
    """
    define custom_tuner
    """

    def __init__(self, **params):
        """
        User have to define their base architecture here and show where is the dataset,
        these params is essential for the tuner to work

        params: model_architecture_path <string> the configure file for architecture
        params: dataset_path <string> the configure file for dataset_path
        params: data_type <string> to define use cnn or rnn

        params: dataset_name <string> to choose to use which image_dir
        params: custom_yaml <string> the config path to the dataset
        """

        #  TODO: We think we can custom image dataset #
        #  TODO:  <18-04-19, VDeamoV> #
        self.model_architecture_path = params["model_architecture_path"]
        self.primitives = params["primitives"]
        self.dataset_path = params["dataset_path"]
        self.dataset_name = params["dataset_name"]
        self.custom_yaml = params["custom_yaml"]
        self.output_path = params["output_path"]

    def update_search_space(self, search_space):
        """
        Must to imply
        """
        print(search_space)

    def receive_trial_result(self, parameter_id, parameters, value):
        '''
        we maybe don't need it either
        '''
        pass

    def generate_parameters(self, parameter_id):
        '''
        we maybe don't need it
        '''
        param = dict({"dataset_path": self.dataset_path,
                      "model_architecture_path": self.model_architecture_path,
                      "primitives": self.primitives,
                      "output_path": self.output_path,
                      "dataset_name": self.dataset_name,
                      "custom_yaml": self.custom_yaml})
        return param
