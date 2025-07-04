# test_oneshot.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from preprocessing import prepare_data_for_models
from main import get_branch_model, build_siamese_model
from utility import *


def remove_class_from_dataset(X, y, classe, label_encoder):
    """
    Rimozione dal dataset X (e dalle etichette y) di tutti i campioni appartenenti alla classe specificata.
    """
    if np.issubdtype(y.dtype, np.integer):
        y_decoded = label_encoder.inverse_transform(y)
    else:
        y_decoded = y

    mask_rimozione = (y_decoded == classe)
    X_rimossi = X[mask_rimozione]
    y_rimossi = y[mask_rimozione]

    mask_filtrata = ~mask_rimozione
    X_filtrato = X[mask_filtrata]
    y_filtrato = y[mask_filtrata]
    return X_filtrato, y_filtrato, X_rimossi, y_rimossi


def train_branch_and_siamese(X_train, y_train_enc, X_val, y_val_enc, genes_len, n_classes):
    """
    Effettua l'addestramento del branch model per la classificazione multi-classe e crea il modello Siamese con il branch congelato.
    """
    # Limita le colonne al numero di geni richiesto
    if X_train.shape[1] > genes_len:
        X_train = X_train.iloc[:, :genes_len]
    if X_val.shape[1] > genes_len:
        X_val = X_val.iloc[:, :genes_len]

    input_shape = (genes_len, 1)
    branch_model = get_branch_model(input_shape=input_shape, n_classes=n_classes)
    branch_model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )

    # Callback per il branch model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    model_checkpoint = ModelCheckpoint(filepath='model/best_branch_model_temp.h5', monitor='val_loss',
                                       save_best_only=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0)
    callbacks = [early_stopping, model_checkpoint, reduce_lr]

    # Preparazione dei dati: reshape e one-hot encoding delle etichette
    X_train_resh = X_train.values.reshape((X_train.shape[0], genes_len, 1))
    X_val_resh = X_val.values.reshape((X_val.shape[0], genes_len, 1))
    y_train_ohe = tf.keras.utils.to_categorical(y_train_enc, num_classes=n_classes)
    y_val_ohe = tf.keras.utils.to_categorical(y_val_enc, num_classes=n_classes)

    branch_model.fit(
        X_train_resh, y_train_ohe,
        validation_data=(X_val_resh, y_val_ohe),
        batch_size=64,
        epochs=20,
        callbacks=callbacks,
        verbose=0
    )

    branch_model.load_weights('model/best_branch_model_temp.h5')
    branch_model.save('model/siamese_branch_temp.h5')

    pretrained_branch = load_model("model/siamese_branch_temp.h5")
    pretrained_branch.trainable = False  # Congela il branch
    siamese_model = build_siamese_model(pretrained_branch, input_shape)
    return branch_model, siamese_model


def main():
    excel_path = "Data.xlsx"
    (x_train_df, x_val_df, x_test_df,
     y_test, y_train_enc, y_val_enc, y_test_enc,
     X_train, X_val, X_test,
     X_train_resh, X_val_resh, X_test_resh,
     y_train_ohe, y_val_ohe, y_test_ohe,
     label_encoder, n_classes, genes_len) = prepare_data_for_models(excel_path)

    label_column = 'CANCER_TYPE'
    print("=== Informazioni sul Test Set ===")
    unique_test_classes = sorted(x_test_df[label_column].unique())
    for cls in unique_test_classes:
        count = len(x_test_df[x_test_df[label_column] == cls])
        print(f"Classe '{cls}': {count} campioni")
    print("=================================\n")

    # Unione del train e validation per la fase di rimozione della classe
    X_train_val = pd.concat([x_train_df, x_val_df], ignore_index=True)
    y_train_val_enc = np.concatenate([y_train_enc, y_val_enc], axis=0)

    classi = sorted(label_encoder.inverse_transform(np.unique(y_train_val_enc)))
    print("Classi nel training+validazione:", classi)

    risultati_finali = []

    for classe_da_rimuovere in classi:
        print(f"\n==============================")
        print(f"RIMOZIONE totale della classe: {classe_da_rimuovere}")
        print(f"==============================\n")

        # Rimozione della classe dal dataset training+validazione
        X_tv_filtrato, y_tv_filtrato, X_tv_rimossi, y_tv_rimossi = remove_class_from_dataset(
            X_train_val, y_train_val_enc, classe_da_rimuovere, label_encoder
        )
        print(f"Dataset training+validazione dopo rimozione: {X_tv_filtrato.shape}")
        print(f"Campioni rimossi per la classe '{classe_da_rimuovere}': {len(X_tv_rimossi)}")

        # Addestramento del branch model e del modello Siamese sul dataset filtrato
        branch_model, siamese_model = train_branch_and_siamese(
            X_tv_filtrato, y_tv_filtrato,
            x_val_df.iloc[:, :genes_len], y_val_enc,
            genes_len, n_classes
        )

        # Preparazione dei dati per l'addestramento del modello Siamese
        if X_tv_filtrato.shape[1] > genes_len:
            X_tv_filtrato = X_tv_filtrato.iloc[:, :genes_len]
        X_tv_filtrato_resh = X_tv_filtrato.values.reshape((X_tv_filtrato.shape[0], genes_len, 1))
        y_tv_filtrato_enc = y_tv_filtrato
        pairs_A_train, pairs_B_train, pair_labels_train = create_siamese_pairs(
            X_tv_filtrato_resh, y_tv_filtrato_enc
        )

        X_val_resh = x_val_df.iloc[:, :genes_len].values.reshape((x_val_df.shape[0], genes_len, 1))
        pairs_A_val, pairs_B_val, pair_labels_val = create_siamese_pairs(X_val_resh, y_val_enc)

        # Callback per il modello Siamese
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
        checkpoint_siamese = ModelCheckpoint('model/best_siamese_temp.h5', monitor='val_loss', save_best_only=True,
                                             verbose=0)
        reduce_lr_siamese = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0)
        siamese_cb = [early_stopping, checkpoint_siamese, reduce_lr_siamese]

        siamese_model.fit(
            [pairs_A_train, pairs_B_train],
            pair_labels_train,
            batch_size=64,
            epochs=20,
            validation_data=([pairs_A_val, pairs_B_val], pair_labels_val),
            callbacks=siamese_cb,
            verbose=0
        )
        siamese_model.load_weights('model/best_siamese_temp.h5')

        # Caricamento del branch pre-addestrato da usare come feature extractor
        pretrained_branch = load_model('model/siamese_branch_temp.h5')
        pretrained_branch.trainable = False

        test_ind = list(x_test_df.index)
        print(f"\n*** Test One-Shot/K-Shot per la classe rimossa '{classe_da_rimuovere}' ***")

        # Esperimento One-Shot con k=1
        acc_k1, preds_detail_k1 = test_kshot(
            model=siamese_model,
            genes_len=genes_len,
            x_test_df=x_test_df,
            test_ind=test_ind,
            X_test_resh=X_test_resh,
            k_shot=1,
            label_column=label_column,
            branch_extractor=pretrained_branch
        )
        print(f"Accuracy One-Shot (k=1): {acc_k1:.2f}%")

        # Esperimento K-Shot con k=5
        acc_k5, preds_detail_k5 = test_kshot(
            model=siamese_model,
            genes_len=genes_len,
            x_test_df=x_test_df,
            test_ind=test_ind,
            X_test_resh=X_test_resh,
            k_shot=5,
            label_column=label_column,
            branch_extractor=pretrained_branch
        )
        print(f"Accuracy K-Shot (k=5): {acc_k5:.2f}%")

        risultati_finali.append({
            "Classe_rimossa": classe_da_rimuovere,
            "OneShot_k1_Accuracy": acc_k1,
            "KShot_k5_Accuracy": acc_k5
        })

    print("\n===== RISULTATI FINALI DEGLI ESPERIMENTI =====")
    for r in risultati_finali:
        print(f"Classe rimossa: {r['Classe_rimossa']}")
        print(f"  One-Shot (k=1) Accuracy: {r['OneShot_k1_Accuracy']:.2f}%")
        print(f"  K-Shot (k=5) Accuracy: {r['KShot_k5_Accuracy']:.2f}%\n")

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_path = os.path.join(results_dir, "One_K_Shot.json")
    with open(results_path, 'w') as f:
        json.dump(risultati_finali, f, indent=4)
    print(f"\nI risultati sono stati salvati in '{results_path}'.")


if __name__ == "__main__":
    main()
