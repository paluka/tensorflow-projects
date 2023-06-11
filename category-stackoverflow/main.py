import tensorflow as tf
import matplotlib.pyplot as plt

batch_size = 32
seed = 42

raw_train_ds, raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    "stack_overflow_16k/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="both",
    seed=seed,
    # shuffle=False, don't need because of provided seed
)


raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    "stack_overflow_16k/test", batch_size=batch_size
)

max_features = 10000
sequence_length = 250

vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_features, output_mode="int", output_sequence_length=sequence_length
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
        tf.keras.layers.Dense(4),
    ]
)

model.summary()

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

epochs = 10
callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)
history = model.fit(
    train_ds, validation_data=val_ds, epochs=epochs, callbacks=[callback]
)

loss, accuracy = model.evaluate(test_ds)

print(f"\nThe loss is {loss}")
print(f"The accuracy is {accuracy * 100}%\n")

history_dict = history.history
history_dict.keys()

acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
loss = history_dict["loss"]
val_loss = history_dict["val_loss"]


plot_epochs = range(1, len(acc) + 1)
plot_data_names = ["loss", "accuracy"]

plt.figure(figsize=(5, 5))

for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.plot(plot_epochs, loss, "b", label=f"Training {plot_data_names[i]}")
    plt.plot(plot_epochs, val_loss, "r", label=f"Validation {plot_data_names[i]}")
    plt.xlabel("Epochs")
    plt.ylabel(plot_data_names[i])
    plt.legend()

plt.show()

export_model = tf.keras.Sequential([vectorize_layer, model])

export_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryFocalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

loss, accuracy = export_model.evaluate(raw_test_ds)
print(f"\nThe loss is {loss}")
print(f"The accuracy is {accuracy * 100}%\n")
