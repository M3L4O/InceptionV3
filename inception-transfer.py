import wandb
import tensorflow as tf
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from tensorflow import keras
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

import pandas as pd
import os


def get_data_iterator(img_size=(224, 224), batch_size=32, mode="binary"):
    train_df = pd.read_csv(TRAIN_FILE)
    validation_df = pd.read_csv(VALIDATION_FILE)
    test_df = pd.read_csv(TEST_FILE)

    gen = ImageDataGenerator(rescale=1.0 / 255)

    train_iter = gen.flow_from_dataframe(
        train_df,
        x_col="image",
        y_col="label",
        target_size=img_size,
        class_mode=mode,
        shuffle=True,
        batch_size=batch_size,
    )

    validation_iter = gen.flow_from_dataframe(
        validation_df,
        x_col="image",
        y_col="label",
        target_size=img_size,
        class_mode=mode,
        shuffle=True,
        batch_size=batch_size,
    )

    test_iter = gen.flow_from_dataframe(
        test_df,
        x_col="image",
        y_col="label",
        target_size=img_size,
        class_mode=mode,
        shuffle=True,
        batch_size=batch_size,
    )

    return train_iter, validation_iter, test_iter


def get_model(learning_rate=0.01):
    """
    Função utilizada para criar o modelo a ser treinado.

    Returns
    -------
    keras.Model
        Modelo a ser treinado

    """
    base_model = InceptionV3(input_shape=(224, 224, 3), include_top=False)
    base_model.trainable = False

    gap = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(1, activation="sigmoid")(gap)

    model = Model(inputs=base_model.input, outputs=predictions)

    adam = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=METRICS)

    return model


def train():
    run = wandb.init()
    config = wandb.config
    batch_size = 32
    learning_rate = 0.01

    train_iter, validation_iter, test_iter = get_data_iterator()
    model = get_model(learning_rate)

    checkpoint = WandbModelCheckpoint(
        os.path.join(MODELS_DIR, TRANSFER_NAME),
        monitor="loss",
        verbose=1,
        save_best_only=True,
    )

    stop = EarlyStopping(
        monitor="loss",
        min_delta=0,
        patience=40,
        verbose=1,
        restore_best_weights=True,
        mode="auto",
    )

    _ = model.fit(
        train_iter,
        epochs=200,
        validation_data=validation_iter,
        callbacks=[checkpoint, stop, WandbMetricsLogger()],
        verbose=1,
    )

    metrics = list(map(lambda x: f"test/{x}", model.metrics_names))
    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, TRANSFER_NAME))
    score = model.evaluate(test_iter)

    results = dict(zip(metrics, score))

    wandb.log(results)
    run.finish()


if __name__ == "__main__":
    wandb.login(key="01638e12d252ba3f829bf5b44b30a59a744ca6d4")
    METRICS = [
        tf.keras.metrics.TruePositives(name="tp"),
        tf.keras.metrics.FalsePositives(name="fp"),
        tf.keras.metrics.TrueNegatives(name="tn"),
        tf.keras.metrics.FalseNegatives(name="fn"),
        tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    TRAIN_FILE = "train.csv"
    VALIDATION_FILE = "validation.csv"
    TEST_FILE = "test.csv"
    TRANSFER_NAME = "InceptionV3_transfer.h5"
    FINE_NAME = "InceptionV3_fine.h5"
    MODELS_DIR = os.path.abspath("./trained_models/")

    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)

    sweep_configuration = {
        "name": "inceptionV3-sweep",
        "entity": "joaovictormelo",
        "method": "grid",
        "metric": {"goal": "maximize", "name": "epoch/val_binary_accuracy"},
        "parameters": {
            "batch_size": {"values": [32]},
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="Inception-Transfer")
    wandb.agent(sweep_id, project="Inception-Transfer", function=train)
