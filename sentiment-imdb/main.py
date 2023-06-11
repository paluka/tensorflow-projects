import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import shutil
import string

# from tensorflow.keras import layers
# from tensorflow.keras import losses

# url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
# dataset = tf.keras.utils.get_file(
#     "aclImdb_v1", url, untar=True, cache_dir=".", cache_subdir=""
# )

# dataset_dir = os.path.join(os.path.dirname(dataset), "aclImdb")

# os.listdir(dataset_dir)

# sample_file = os.path.join(dataset_dir, "train", "pos/1181_9.txt")

# with open(sample_file) as file:
#     print(file.read())

# remove_dir = os.path.join(dataset_dir, "train", "unsup")
# shutil.rmtree(remove_dir)

batch_size = 32
seed = 42

raw_train_ds, raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="both",
    seed=seed,
)

# for text_batch, label_batch in raw_train_ds.take(1):
#     for i in range(3):
#         print("Review", text_batch.numpy()[i])
#         print("Label", label_batch.numpy()[i])


raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size
)


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )


max_features = 10000
sequence_length = 250

vectorize_layer = tf.keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

embdedding_dim = 16

model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(max_features + 1, embdedding_dim),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ]
)

model.summary()

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=tf.metrics.BinaryAccuracy(threshold=0.0),
)

epochs = 10
callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)
history = model.fit(
    train_ds, validation_data=val_ds, epochs=epochs, callbacks=[callback]
)

loss, accuracy = model.evaluate(test_ds)

print(loss, accuracy)

history_dict = history.history
history_dict.keys()

acc = history_dict["binary_accuracy"]
val_acc = history_dict["val_binary_accuracy"]
loss = history_dict["loss"]
val_loss = history_dict["val_loss"]


plot_epochs = range(1, len(acc) + 1)
plot_data_names = ["loss", "accuracy"]

plt.figure(figsize=(1, 1))

for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.plot(plot_epochs, loss, "b", label=f"Training {plot_data_names[i]}")
    plt.plot(plot_epochs, val_loss, "r", label=f"Validation {plot_data_names[i]}")
    plt.xlabel("Epochs")
    plt.ylabel(plot_data_names[i])
    plt.legend()

plt.show()


export_model = tf.keras.Sequential(
    [vectorize_layer, model, tf.keras.layers.Activation("sigmoid")]
)

export_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryFocalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

loss, accuracy = export_model.evaluate(raw_test_ds)
print(loss, accuracy)

new_inferences = ["The movie was great!", "The movie was okay.", "The movie sucked"]

predictions = export_model.predict(new_inferences)
print(predictions)
