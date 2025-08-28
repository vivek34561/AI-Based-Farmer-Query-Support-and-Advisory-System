import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

#     Here’s why we don’t use .pkl for deep learning models:

# 🔹 1. Pickle is Python-specific

# .pkl is based on Python’s pickle module (serialization).

# It only works inside Python, with the same version of libraries.

# Keras/TensorFlow models are more complex (contain architecture, weights, training config, optimizer state).

# Pickle struggles to serialize and restore complex objects like layers/graphs.
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    # Instead of building a new model every time, you just load the base model you trained or updated previously.
    def train_valid_generator(self):
#         What it does: Creates data generators for training and validation.

# Why: Instead of loading all images at once (which doesn’t fit in memory), Keras uses generators to load batches of images during training.

        datagenerator_kwargs = dict(rescale = 1./255 , validation_split=0.20)
# rescale=1./255

# Raw image pixels are integers in [0, 255].

# Dividing by 255 normalizes them to [0, 1].

# Normalization helps neural networks train faster and more stably.

# validation_split=0.20

# Splits dataset into 80% training and 20% validation.

# This way, from the same folder of images, you can generate both training and validation datasets.
        dataflow_kwargs = dict(target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        # target_size=self.config.params_image_size[:-1]

# Resizes all images to the same shape.

# Example: if params_image_size = [224, 224, 3], then [:-1] = (224, 224) (height × width).

# Ensures input images match the model’s expected input size.
# Number of images processed at once before updating weights.

# Example: batch_size=32 → model trains on 32 images at a time.

# interpolation="bilinear"

# Method used when resizing images.

# "bilinear" = smoother than nearest neighbor, good for natural images.
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator( **datagenerator_kwargs)

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )
# Loads validation images from folders (CT-KIDNEY-DATASET...)

# Uses 20% of the dataset.

# Does not shuffle (so evaluation is stable).
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
#             If augmentation is enabled, it applies transformations (rotation, flipping, zoom, etc.) to make the model more robust.

# If augmentation is disabled, it just uses the same settings as validation (no transformations).
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )
        print("Class Indices:", self.train_generator.class_indices)
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



    
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
# self.train_generator.samples: total training images.

# self.train_generator.batch_size: images per batch.

# // is floor division. It drops any partial (incomplete) last batch.

# If you have 2,300 images and batch size 32:

# 2300 // 32 = 71 (since 71×32 = 2272). You skip the last 28 images.

# Same logic for validation.

# ⚠️ Because // discards the remainder, you may not train/evaluate on all images each epoch. Usually you want to include that last partial batch.
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )
#         Trains for params_epochs epochs.

# Each epoch uses steps_per_epoch batches from train_generator.

# After each epoch, it runs validation_steps batches from valid_generator to compute val metrics.

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
#         Saves the trained model.

# If trained_model_path ends with .h5 → saves HDF5 file.

# If it’s a folder path (no extension) → saves a TensorFlow SavedModel directory.