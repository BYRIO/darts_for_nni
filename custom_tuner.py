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

    def __init__(self, model_architecture_path, dataset_path, data_type):
        """
        User have to define their base architecture here and show where is the dataset,
        these params is essential for the tuner to work

        params: model_architecture_path <string> the configure file for architecture
        params: dataset_path <string> the configure file for dataset_path
        params: data_type <string> to define use cnn or rnn


        """
        #  TODO: We think we can custom image dataset #
        #  TODO:  <18-04-19, VDeamoV> #
        self.model_architecture_path = model_architecture_path
        self.dataset_path = dataset_path
        self.data_type = data_type

    def update_search_space(self, search_space):
        """
        Must to imply
        """
        print(search_space)
        print("fuck it")

    def receive_trial_result(self, parameter_id, parameters, value):
        '''
        we maybe don't need it either
        '''
        pass

    def generate_parameters(self, parameter_id):
        '''
        we maybe don't need it
        '''
        pass
