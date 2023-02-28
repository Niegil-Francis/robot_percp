# VR based benchmarking of Fast Target Prediction Algorithms

This respository contains the data and target prediction models used for the final project of the robot perception course at NYU. 

The repository also contains the VR environment developed on unity and interfaced with the occulus rift 2 headset mounted with the leap motion camera.

The folders are sturctured as shown below: <br>
1.  The folder [all_data](./all_data/) contains the data for the target prediction - both 3D target point prediction and target classification.
    - The folder [classification_1](./all_data/sep_1/) contains the data for the first target separation.
2. The folder [models](./models/) contains all the target prediction models.

The notebook [model_training.ipynb](./model_training.ipynb) is used for model training and [model_prediction.ipynb](./model_prediction.ipynb) is used for the predictions that are interfaced with unity.
