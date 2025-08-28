import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import os
from dotenv import load_dotenv

from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories, save_json



# During Training:

# You already created a train_generator and a valid_generator.

# The model used them for training (fit() method).

# 🔹 During Evaluation:

# You are loading a saved model from disk (self.load_model(...)).

# This model does not store your dataset or generator information — it only stores the trained weights and architecture.

# So, to test or evaluate the model again, you must recreate the validation generator that knows:

# where your validation images are (directory=self.config.training_data)

# how to resize them (target_size=...)

# how to normalize them (rescale=1./255)

# and which subset to use (subset="validation")
# 👉 In short:
# _valid_generator() is used again because the model only saves weights & architecture, not the dataset pipeline.
# To evaluate, we must recreate the validation generator and feed data into the model again.
class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1. / 255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()
# Load trained model.

# Create validation generator.

# Evaluate model on validation data (loss, accuracy).

# Save scores. self.score = [0.45, 0.87]
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
# Saves evaluation results into scores.json.

# Example scores.json:
#     {
#   "loss": 0.4521,
#   "accuracy": 0.8712
# }
    def log_into_mlflow(self):
        # Load environment variables from .env
        load_dotenv()

        # Set MLflow tracking URI from env
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri is None:
            raise ValueError("MLFLOW_TRACKING_URI not found in environment variables.")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("kidney_disease_classification")  # Optional: change to "Default" if issues

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )

            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")
# MLflow setup

# Loads tracking URI from .env:

# MLFLOW_TRACKING_URI=http://127.0.0.1:5000


# Sets experiment name: "kidney_disease_classification".

# Logging

# mlflow.log_params(self.config.all_params) → logs model hyperparameters (epochs, batch size, etc.).

# mlflow.log_metrics({...}) → logs performance (loss & accuracy).

# Model storage

# If MLflow server is remote/cloud → registers model with name "VGG16Model".

# If using local file store → just logs locally.

# 📌 Why MLflow?
# It allows:

# Tracking experiments.

# Comparing models.

# Versioning & deployment of models.