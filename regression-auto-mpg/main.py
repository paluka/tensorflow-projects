import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

np.set_printoptions(precision=3, suppress=True)

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = [
    "MPG",
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "Model Year",
    "Origin",
]

raw_dataset = pd.read_csv(
    url, names=column_names, na_values="?", comment="\t", sep=" ", skipinitialspace=True
)

dataset = raw_dataset.copy()

# print(dataset.tail())
# print(dataset.isna().sum())

dataset = dataset.dropna()

dataset["Origin"] = dataset["Origin"].map({1: "USA", 2: "Europe", 3: "Japan"})

# print(dataset["Origin"].tail())

dataset = pd.get_dummies(
    dataset, columns=["Origin"], prefix="", prefix_sep="", dtype=int
)

# print(dataset.tail())

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# sns.pairplot(
#     train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde"
# )

# plt.show()

# print(train_dataset.describe().transpose())

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop("MPG")
test_labels = test_features.pop("MPG")


# first = np.array(train_features[:1])

# with np.printoptions(precision=2, suppress=True):
#     print("First example", first)
#     print()
#     print("Normalized:", normalizer(first).numpy())

####################
# Linear regression with one variable
####################

horsepower = np.array(train_features["Horsepower"])

horsepower_normalizer = tf.keras.layers.Normalization(
    input_shape=[
        1,
    ],
    axis=None,
)

horsepower_normalizer.adapt(horsepower)

horsepower_model = tf.keras.Sequential(
    [horsepower_normalizer, tf.keras.layers.Dense(1)]
)

# print(horsepower_model.summary())

horsepower_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss="mean_absolute_error"
)

history = horsepower_model.fit(
    train_features["Horsepower"],
    train_labels,
    epochs=100,
    verbose=0,
    validation_split=0.2,
)

# hist = pd.DataFrame(history.history)
# hist["epoch"] = history.epoch
# print(hist.tail())


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.ylim([0, 10])
    plt.xlabel("Epoch")
    plt.ylabel("Error [MPG]")
    plt.legend()
    plt.grid(True)
    plt.show()


# plot_loss(history)

test_results = {}

test_results["horsepower_model"] = horsepower_model.evaluate(
    test_features["Horsepower"], test_labels, verbose=0
)

x = tf.linspace(0, 250, 251)
y = horsepower_model.predict(x)


def plot_horsepower(x, y):
    plt.scatter(train_features["Horsepower"], train_labels, label="Data")
    plt.plot(x, y, color="k", label="Predictions")
    plt.xlabel("Horsepower")
    plt.ylabel("MPG")
    plt.legend()
    plt.show()


# plot_horsepower(x, y)

####################
# Linear regression with multiple inputs
####################

normalizer = tf.keras.layers.Normalization(axis=-1)

normalizer.adapt(np.array(train_features))

normalizer.mean.numpy()

linear_model = tf.keras.Sequential([normalizer, tf.keras.layers.Dense(units=1)])

# print(linear_model.layers[1].kernel)

linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss="mean_absolute_error"
)

history = linear_model.fit(
    train_features, train_labels, epochs=100, verbose=0, validation_split=0.2
)

# plot_loss(history)

test_results["linear_model"] = linear_model.evaluate(
    test_features, test_labels, verbose=0
)

####################
# Regression with a deep neural network (DNN)
####################


def build_and_compile_model(norm):
    model = tf.keras.Sequential(
        [
            norm,
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(loss="mean_absolute_error", optimizer=tf.keras.optimizers.Adam(0.001))
    return model


dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)
# print(dnn_horsepower_model.summary())

history = dnn_horsepower_model.fit(
    train_features["Horsepower"],
    train_labels,
    validation_split=0.2,
    verbose=0,
    epochs=100,
)

# plot_loss(history)

x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)

# plot_horsepower(x, y)

test_results["dnn_horsepower_model"] = dnn_horsepower_model.evaluate(
    test_features["Horsepower"], test_labels, verbose=0
)

####################
# Regression using a DNN and multiple inputs
####################

dnn_model = build_and_compile_model(normalizer)
# print(dnn_model.summary())

history = dnn_model.fit(
    train_features, train_labels, validation_split=0.2, verbose=0, epochs=100
)

# plot_loss(history)

test_results["dnn_model"] = dnn_model.evaluate(test_features, test_labels, verbose=0)

# print(pd.DataFrame(test_results, index=["Mean absolute error [MPG]"]).T)

test_predictions = dnn_model.predict(test_features).flatten()

# a = plt.axes(aspect="equal")
# plt.scatter(test_labels, test_predictions)
# plt.xlabel("True Values [MPG]")
# plt.ylabel("Predictions [MPG]")
# lims = [0, 50]
# plt.xlim(lims)
# plt.ylim(lims)
# plt.plot(lims, lims)
# plt.show()


# error = test_predictions - test_labels
# plt.hist(error, bins=25)
# plt.xlabel("Prediction Error [MPG]")
# plt.ylabel("Count")
# plt.show()

# dnn_model.save("dnn_model")

reloaded = tf.keras.models.load_model("dnn_model")

test_results["reloaded"] = reloaded.evaluate(test_features, test_labels, verbose=0)

print(pd.DataFrame(test_results, index=["Mean absolute error [MPG]"]).T)
