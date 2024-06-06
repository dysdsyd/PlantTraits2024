# PlantTraits2024
Welcome to the PlantTraits2024 repository! This project won 1st place in the PlantTraits2024 contest. Below are detailed instructions to set up, train, and customize the project. Big thanks to the authors of all kernels & posts, which were of great inspiration and some features were derived based on them.  
Kaggle Profile : [Daft Vader](https://www.kaggle.com/syeddanish)

<p align="center">
  <img src="https://github.com/dysdsyd/PlantTraits2024/assets/9487316/7131c90c-6dac-4536-8fe6-f55532fb029c" alt="Hydra-like Structure" width="200" height="200" style="border-radius: 50%; object-fit: cover;">
</p>

## Installation
#### Step 1 - Install Your Virtual Environment Manager
Install your preferred environment manager. The example shown below uses Miniconda:

1. Download Miniconda:
    ```sh
    # Find the latest URL here: https://docs.conda.io/en/latest/miniconda.html
    wget <miniconda_release_url>
    sh <downloaded_conda_file>
    ```

#### Step 2 - Create the Environment
Set up your environment using Conda:

1. Create a new conda environment with Python 3.10:
    ```sh
    conda create --name my_env python=3.10
    ```
2. Activate the environment:
    ```sh
    conda activate my_env
    ```
3. Install the required packages from `requirements.txt`:
    ```sh
    pip install -r requirements.txt
    ```
4. Install the project in editable mode:
    ```sh
    pip install -e .
    ```

## Model Training

### Step 1 - Data Setup
Follow the instructions in the setup notebook:
- Open and run `notebooks/plantTraits/data_setup.ipynb` to prepare the data.

### Step 2 - Start Training
Run the following command to start training:
```sh
python src/fgvc/train.py experiment=plant_traits description="test"
```

### Step 3 - Evaluate on Test Set
Use the evaluation notebook:
- Open and run `notebooks/plantTraits/run_submission.ipynb` to evaluate the model on the test set.

## Customizing Your Training

To customize various aspects of the training process, modify the following configuration files:

1. **Dataset Customization:** Modify `configs/data/plant_traits_data.yaml` to customize the dataset parameters.
2. **Model Customization:** Modify `configs/model/plant_traits_model.yaml` to adjust model configurations.
3. **Experiment Customization:** Modify `configs/experiment/plant_traits.yaml` to change overall experiment settings.

### Modifying Source Code
1. **Dataset Modification:** Modify `src/fgvc/data/plant_traits_data.py`
2. **Model Customization:** Modify `src/fgvc/models/plant_traits_model.py`

## Acknowledgement
This experimentation template is generated from [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template). For more details about the template, please refer to their documentation.

Thank you for using PlantTraits2024! If you have any questions or encounter issues, please feel free to open an issue or contact the maintainers.
