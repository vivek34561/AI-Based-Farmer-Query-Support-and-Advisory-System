from dataclasses import dataclass
from pathlib import Path




# The decorator @dataclass(frozen=True) is used to automatically create a class for storing data, with some extra features. Let me break it down for your DataIngestionConfig example:

#  What @dataclass does

# Automatically generates boilerplate code for the class, like:

# __init__() → to initialize attributes

# __repr__() → for easy printing

# __eq__() → for comparison

# Without @dataclass, you’d have to manually write:


# class DataIngestionConfig:
#     def __init__(self, root_dir, source_URL, local_data_file, unzip_dir):
#         self.root_dir = root_dir
#         self.source_URL = source_URL
#         self.local_data_file = local_data_file
#         self.unzip_dir = unzip_dir
# What frozen=True does

# Makes the class immutable.

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path



@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int



@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list



@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int