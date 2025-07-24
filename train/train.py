# %% [markdown]
# # Dataset loading and preparation

# %%
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.preprocessing
import os

def load_dataset(dataset_name):
    if dataset_name == "songs":
        # https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks/data
        full_dataset = pd.read_csv("data/spotify_data.csv", index_col=0)
        full_dataset.rename(columns={"genre": "label"}, inplace=True)
        full_dataset["label"] = sklearn.preprocessing.LabelEncoder().fit_transform(full_dataset["label"])
        X = full_dataset[["key", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "time_signature", "popularity"]]
        X = pd.get_dummies(X, columns=["key", "time_signature"], drop_first=True)
        y = full_dataset["label"]
    elif dataset_name == "covertype":
        # https://archive.ics.uci.edu/dataset/31/covertype
        full_dataset = sklearn.datasets.fetch_covtype(as_frame=True).frame
        full_dataset.rename(columns={"Cover_Type": "label"}, inplace=True)
        full_dataset["label"] = full_dataset["label"] - 1
        X = full_dataset.drop(columns=["label"])
        y = full_dataset["label"]
    elif dataset_name == "nids":
        # https://rdm.uq.edu.au/files/2ad93cd0-ef9c-11ed-827d-e762de186848
        # https://staff.itee.uq.edu.au/marius/NIDS_datasets/ 
        full_dataset = pd.read_csv("data/NF-ToN-IoT.csv").drop(columns=["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label"])
        full_dataset.rename(columns={"Attack": "label"}, inplace=True)
        X = full_dataset.drop(columns=["label"])
        le = sklearn.preprocessing.LabelEncoder()
        y = le.fit_transform(full_dataset["label"])
    elif dataset_name == "urls":
        # https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
        full_dataset = pd.read_csv("data/PhiUSIIL_Phishing_URL_Dataset.csv")
        url_based_features = ["URLLength", "IsDomainIP", "CharContinuationRate", "TLDLength", "NoOfSubDomain", "NoOfObfuscatedChar", "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS"]
        X = full_dataset[url_based_features]
        y = full_dataset["label"]
    elif dataset_name.startswith("gaussian"):
        classes = 8 
        n = 100000000
        distance = 2
        sigma = [0.25, 0.5, 0.75, 1.0][int(dataset_name[-1])]
        np.random.seed(42)
        X = []
        y = []
        for i in range(classes):
            mean = i * distance
            X.append(np.random.normal(mean, sigma, n // classes))
            y.append(np.full(n // classes, i))
        full_dataset = pd.DataFrame({"X": np.concatenate(X), "label": np.concatenate(y)})
        X = full_dataset.drop(columns=["label"]).values.reshape(-1, 1)
        y = full_dataset["label"].values
    else:
        raise ValueError("Invalid dataset name")

    value_counts = full_dataset["label"].value_counts()
    frequecies = value_counts / value_counts.sum()
    entropy = -np.sum(frequecies * np.log2(frequecies))
    classes = len(value_counts)
    print(f"Dataset: {dataset_name}")
    print(f"Number of samples: {len(full_dataset)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {classes}")
    print(f"Entropy: {entropy:.4f}")

    #X.hist(bins=50, figsize=(15, 10))
    print(full_dataset["label"].value_counts())
    return X, y, classes

# %%
def prepare_dataset(dataset_name, X, y, classes):
    from sklearn.model_selection import train_test_split
    import struct
    import base64
    sc = sklearn.preprocessing.StandardScaler()
    Xt = sc.fit_transform(X).astype("float32")
    X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2, random_state=42, stratify=y)
    os.makedirs("processed_data/data_sux4j", exist_ok=True)

    # Write data for learned retrieval
    np.insert(y, 0, classes).astype(np.uint16).tofile(f"processed_data/{dataset_name}_y.lrbin")
    with open(f"processed_data/{dataset_name}_X.lrbin", "wb") as f:
        f.write(struct.pack("<Q", Xt.shape[0]))
        f.write(struct.pack("<Q", Xt.shape[1]))
        Xt.tofile(f)

    # Write data for sux4j
    np.array(y, dtype=np.int64).byteswap().tofile(f"processed_data/data_sux4j/{dataset_name}_y.sux4j")
    with open(f"processed_data/data_sux4j/{dataset_name}_X.sux4j", "wb") as f:
        for i, row in enumerate(Xt):
            byte_data = row.tobytes().replace(b"\0", b" ").replace(b"\n", b" ").replace(b"\r", b"")
            byte_data += base64.b85encode(i.to_bytes(4, "big"))
            f.write(byte_data)
            f.write(b"\n")
    
    return X_train, X_test, y_train, y_test

# %% [markdown]
# # Model training

# %%
def train(model_path, dataset_name, X_train, y_train, classes, num_layers, hidden_units):
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import keras
    import time
    import json
    keras.utils.set_random_seed(42)

    model_name = f"{dataset_name}_mlp_L{num_layers}_H{hidden_units}"
    filename = f"{model_name}_{time.strftime('%Y%m%d-%H%M%S')}"

    if classes > 2:
        output_layer = keras.layers.Dense(classes, activation="softmax")
        learning_rate = 1e-3
        monitor = "val_sparse_categorical_accuracy"
        loss = "sparse_categorical_crossentropy"
        metrics = [keras.metrics.SparseCategoricalAccuracy(),
                keras.metrics.SparseTopKCategoricalAccuracy(3, name="sparse_top_3_accuracy")]
    else:
        output_layer = keras.layers.Dense(1, activation="sigmoid")
        learning_rate = 1e-3
        monitor = "val_binary_accuracy"
        loss = "binary_crossentropy"
        metrics = [keras.metrics.BinaryAccuracy()]

    model = keras.models.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],), name="input"),
        *[keras.layers.Dense(hidden_units, activation="relu") for _ in range(num_layers)],
        output_layer,
    ])

    es = keras.callbacks.EarlyStopping(
        monitor=monitor,
        verbose=1,
        patience=10,
        min_delta=0.00001,
        restore_best_weights=True,
    )

    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=metrics,
    )

    print(filename)
    model.summary()

    train_start = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=128,
        verbose=2,
        validation_split=0.1,
        callbacks=[es],
    )
    train_end = time.perf_counter()
    training_seconds = train_end - train_start
    model_params = model.count_params()
    print(f"Training time: {training_seconds} s")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(f"{model_path}/{filename}.keras")
    with open(f"{model_path}/{filename}.json", "w") as f:
        info = {"training_seconds": training_seconds,
                "model_params": model_params,
                "model_l": num_layers,
                "model_h": hidden_units}
        json.dump(history.history | info, f)
    return filename, info

# %% [markdown]
# # Export to TFLite and evaluate

# %%
def export_tflite(model_path, filename, X_train, X_test, y_test, classes, info):
    import tensorflow as tf
    import keras
    from tensorflow.lite.python import interpreter

    def representative_dataset():
        for x in X_train[:1000]:
            yield {"input": x}

    quantized_modes = []

    for quantization in ["uint8", "float16", "float32"]:
        print("#" * 35, f"{quantization} quantization", "#" * 35)
        tf.config.set_visible_devices([], "GPU")
        model = keras.saving.load_model(f"{model_path}/{filename}.keras", compile=False)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        if quantization == "uint8":
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.representative_dataset = representative_dataset
        elif quantization == "float16":
            converter.target_spec.supported_types = [tf.float16]
            
        tflite_model = converter.convert()
        with open(f"{model_path}/{filename}_{quantization}.tflite", "wb") as f:
            f.write(tflite_model)
        quantized_modes.append(tflite_model)


    for quantization, tflite_model in zip(["uint8", "float16", "float32"], quantized_modes):
        interp = interpreter.InterpreterWithCustomOps(model_content=tflite_model)
        interp.allocate_tensors()
        model_runner = interp.get_signature_runner("serving_default")
        tflite_y = model_runner(input=X_test)
        tflite_y = list(tflite_y.values())[0]
        if classes == 2:
            tflite_y = np.array([1 - tflite_y, tflite_y]).T
        accuracy = keras.metrics.sparse_categorical_accuracy(y_test, tflite_y).numpy().mean()
        top3accuracy = keras.metrics.sparse_top_k_categorical_accuracy(y_test, tflite_y, k=3).numpy().mean()
        print(f"{quantization} quantization file bytes: {len(tflite_model)}")
        print(f"{quantization} quantization accuracy: {accuracy}")

        with open(f"{model_path}/{filename}_{quantization}.tflite_eval.txt", "w") as f:
            f.write(f"model_l={info['model_l']} ")
            f.write(f"model_h={info['model_h']} ")
            f.write(f"quant={quantization} ")
            f.write(f"training_seconds={info['training_seconds']} ")
            f.write(f"model_params={info['model_params']} ")
            f.write(f"test_accuracy={accuracy*100} ")
            if classes > 2:
                f.write(f"test_top3_accuracy={top3accuracy*100}")

# %% [markdown]
# # Run training and export for all datasets

# %%
for dataset_name in ["songs", "covertype", "nids", "urls", "gaussian0", "gaussian1", "gaussian2", "gaussian3"]:
    print(f"Processing dataset: {dataset_name}")
    X, y, classes = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = prepare_dataset(dataset_name, X, y, classes)
    if dataset_name.startswith("gaussian"):
        continue
    for num_layers, hidden_units in [(0, 0), (1, 50), (1, 100), (2, 50)]:    
        print(f"Training model with {num_layers} layers and {hidden_units} hidden units on {dataset_name}")
        model_path = f"models/{dataset_name}_models/"
        filename, info = train(model_path, dataset_name, X_train, y_train, classes, num_layers, hidden_units)
        export_tflite(model_path, filename, X_train, X_test, y_test, classes, info)

# %%



