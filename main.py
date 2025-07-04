# main.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Precision is ill-defined and being set to 0.0 in labels with no predicted samples")
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dropout, Dense)
from tensorflow.keras.models import Model

from utility import *
from preprocessing import prepare_data_for_models


def get_branch_model(input_shape, n_classes=9):
    """
    Branch della rete siamese per classificazione multi-classe.
    """
    # Input del modello
    inp = Input(shape=input_shape)

    # Regolarizzazione: L2 con weight decay pari a 1e-5
    regularizer = tf.keras.regularizers.l2(1e-5)

    # Primo blocco convoluzionale
    x = Conv1D(filters=32, kernel_size=5, activation='relu',
               padding='same', kernel_regularizer=regularizer)(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Secondo blocco convoluzionale
    x = Conv1D(filters=256, kernel_size=3, activation='relu',
               padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Terzo blocco convoluzionale
    x = Conv1D(filters=256, kernel_size=3, activation='relu',
               padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Flatten e strati fully connected
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizer)(x)
    x = Dropout(0.4)(x)

    # Output: layer finale con n_classes e attivazione softmax per la classificazione
    output = Dense(n_classes, activation='softmax', name="output_layer",
                   kernel_regularizer=regularizer)(x)

    model = Model(inputs=inp, outputs=output, name="SiameseBranch")
    return model


def build_siamese_model(pretrained_branch, input_shape):
    """
    Costruzione e compilazione del modello Siamese.
    """
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # Calcolo degli embedding con il branch pre-addestrato
    encoded_a = pretrained_branch(input_a)
    encoded_b = pretrained_branch(input_b)

    # Calcolo della distanza L2
    distance_output = last_layer(encoded_a, encoded_b, distance='L2')  #cosine, L1, L2

    # Aggiunta di uno strato finale per mappare la distanza in una probabilità [0,1]
    prediction = Dense(1, activation='sigmoid')(distance_output)

    # Creazione e compilazione del modello Siamese
    siamese_model = Model(inputs=[input_a, input_b], outputs=prediction)
    siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return siamese_model


if __name__ == "__main__":
    excel_path = "Data.xlsx"
    (x_train_df, x_val_df, x_test_df,
     y_test, y_train_enc, y_val_enc, y_test_enc,
     X_train, X_val, X_test,
     X_train_resh, X_val_resh, X_test_resh,
     y_train_ohe, y_val_ohe, y_test_ohe,
     le, n_classes, genes_len) = prepare_data_for_models(excel_path)

    # Definizione degli indici di test e dello shape dell'input
    test_ind = list(x_test_df.index)
    input_shape = (genes_len, 1)

    # Creazione del branch model per la classificazione multi-classe
    branch_model = get_branch_model(input_shape=input_shape, n_classes=n_classes)
    branch_model.compile(loss='categorical_crossentropy',
                         optimizer=Adam(learning_rate=0.0001),
                         metrics=['accuracy'])

    # Definizione dei callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        filepath='model/best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    callbacks = [early_stopping, model_checkpoint, reduce_lr]

    # Training del branch model
    history = branch_model.fit(
        X_train_resh, y_train_ohe,
        validation_data=(X_val_resh, y_val_ohe),
        batch_size=64,
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )
    plot_training(history, "results/branch_model_training.png")

    test_loss, test_accuracy = branch_model.evaluate(X_test_resh, y_test_ohe, verbose=0)
    print(f"\nAccuracy sul test (Branch): {test_accuracy * 100:.2f}%")

    branch_model.save("model/siamese_branch_pretrained.h5")

    # Caricamento del modello pre-addestrato e blocco dei pesi se lo si vuole usare come feature extractor
    pretrained_branch = load_model("model/siamese_branch_pretrained.h5")
    pretrained_branch.trainable = False  # Blocco dei pesi

    # Costruzione del modello Siamese tramite la funzione dedicata
    siamese_model = build_siamese_model(pretrained_branch, input_shape)

    # Creazione delle coppie per il training dal set di training
    pairs_A_train, pairs_B_train, pair_labels_train = create_siamese_pairs(X_train_resh, y_train_enc)

    # Creazione delle coppie per la validazione dal set di validazione
    pairs_A_val, pairs_B_val, pair_labels_val = create_siamese_pairs(X_val_resh, y_val_enc)

    # Training del modello Siamese sulle coppie
    siamese_history = siamese_model.fit(
        [pairs_A_train, pairs_B_train],
        pair_labels_train,
        batch_size=64,
        epochs=20,
        validation_data=([pairs_A_val, pairs_B_val], pair_labels_val),
        callbacks=callbacks,
        verbose=1
    )
    plot_training(siamese_history, "results/siamese_model_training.png")

    # Accuracy one-shot per il modello siamese
    acc_tumor, preds_detail_tumor = test_kshot(
            model=siamese_model,
            genes_len=genes_len,
            x_test_df=x_test_df,
            test_ind=test_ind,
            X_test_resh=X_test_resh,
            k_shot=5,
            label_column='CANCER_TYPE',
            branch_extractor=pretrained_branch
        )
    print(f"\n[One-shot] Accuratezza One-Shot (Siamese): {acc_tumor:.2f}%")

    # Salvataggio dei risultati in un file csv
    df_preds_detail = pd.DataFrame(preds_detail_tumor, columns=['index', 'true', 'predicted'])
    csv_filepath = "results/preds_detail_tumor.csv"
    df_preds_detail.to_csv(csv_filepath, index=False)
    print(f"Dettagli delle predizioni sono salvati in: {csv_filepath}")

    # Estrazione delle etichette vere e previste per la matrice di confusione e per il report di classificazione
    y_true = [true for _, true, _ in preds_detail_tumor]
    y_pred = [pred for _, _, pred in preds_detail_tumor]
    classes = sorted(x_test_df['CANCER_TYPE'].unique())

    # Calcolo e salvataggio dela Confusion Matrix per il modello Siamese
    plot_and_save_confusion_matrix(y_true, y_pred, classes, filename="results/siamese_confusion_matrix.png")

    # Generazione del report esteso che include anche specificity e sensitivity
    report = extended_classification_report(y_true, y_pred, classes)
    print(report)

    # Salvataggio del report in formato txt
    report_siamese_filepath = "results/classification_report_siamese.txt"
    with open(report_siamese_filepath, "w") as f:
        f.write(report)
    print(f"Report delle metriche salvato come '{report_siamese_filepath}'.")

    # Salvataggio delle seguenti metriche in formato json
    all_metrics = {
        "branch_model": {
            "test_accuracy": test_accuracy,
        },
        "siamese_model": {
            "one_shot_accuracy": acc_tumor
        }
    }
    metrics_json_filepath = os.path.join("results/all_metrics.json")
    with open(metrics_json_filepath, "w") as json_file:
        json.dump(all_metrics, json_file, indent=4)
    print(f"Tutte le metriche sono salvate in {metrics_json_filepath}.")
