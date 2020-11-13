from typing import Dict
from typing import List

import tensorflow as tf


def custom_model(train_dataset: tf.data.Dataset,
                eval_dataset: tf.data.Dataset,
                schema: Dict,
                log_dir: str,
                batch_size: int = 32,
                lr: float = 0.0001,
                epochs: int = 10,
                dropout_chance: int = 0.2,
                loss: str = 'mse',
                metrics: List[str] = None,
                hidden_layers: List[int] = None,
                hidden_activation: str = 'relu',
                last_activation: str = 'sigmoid',
                input_units: int = 11,
                output_units: int = 1,
                ):
    """
    returns: a Tensorflow/Keras model
    """
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    eval_dataset = eval_dataset.batch(batch_size, drop_remainder=True)

    if metrics is None:
        metrics = []

    if hidden_layers is None:
        hidden_layers = [64, 32, 16]

    input_layer = tf.keras.layers.Input(shape=(input_units,))

    d = input_layer
    for size in hidden_layers:
        d = tf.keras.layers.Dense(size, activation=hidden_activation)(d)
        d = tf.keras.layers.Dropout(dropout_chance)(d)

    # Assuming that there is only one label
    label_name = list(train_dataset.element_spec[1].keys())[0]

    output_layer = tf.keras.layers.Dense(output_units,
                                         activation=last_activation,
                                         name=label_name)(d)

    model = tf.keras.Model(inputs=input_layer,
                           outputs=output_layer)

    model.compile(loss=loss,
                  optimizer=tf.keras.optimizers.Adam(lr=lr),
                  metrics=metrics)

    model.summary()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=eval_dataset,
        callbacks=[tensorboard_callback])

    return model
