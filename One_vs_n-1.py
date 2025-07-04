# One_vs_n-1.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

from preprocessing import prepare_data_for_models
from main import get_branch_model, build_siamese_model


def remove_class_from_dataset(X, y, classe, label_encoder):
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

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    model_checkpoint = ModelCheckpoint(filepath='model/best_branch_model_temp.h5', monitor='val_loss',
                                       save_best_only=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0)
    callbacks = [early_stopping, model_checkpoint, reduce_lr]

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
    pretrained_branch.trainable = False
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
    print("=== Informazioni sul Dataset Originale ===")
    for cls in sorted(x_test_df[label_column].unique()):
        count = len(x_test_df[x_test_df[label_column] == cls])
        print(f"Classe '{cls}': {count} campioni nel test set")
    print("=================================\n")

    # Combino training e validation set
    X_train_val = pd.concat([x_train_df, x_val_df], ignore_index=True)
    y_train_val_enc = np.concatenate([y_train_enc, y_val_enc], axis=0)
    classi = sorted(label_encoder.inverse_transform(np.unique(y_train_val_enc)))
    print("Classi nel training+validazione:", classi)

    risultati = []

    for classe_target in classi:
        if classe_target.lower() == "normal": continue

        print(f"\n========== ESPERIMENTO per la classe target: {classe_target} ==========")

        # Rimozione della classe target dal set di training e validazione
        X_tv_filtrato, y_tv_filtrato, X_tv_rimossi, y_tv_rimossi = remove_class_from_dataset(
            X_train_val, y_train_val_enc, classe_target, label_encoder
        )

        print(f"Dataset training+validazione dopo rimozione della classe '{classe_target}': {X_tv_filtrato.shape}")
        print(f"Campioni rimossi per '{classe_target}': {len(X_tv_rimossi)}")

        # Rimozione della classe target anche dal test set per addestrare il modello
        X_test_filtrato_df = x_test_df[x_test_df[label_column] != classe_target].copy()
        X_test_target_df = x_test_df[x_test_df[label_column] == classe_target].copy()

        if len(X_test_target_df) == 0:
            print(f"Non sono presenti campioni nel test set per la classe '{classe_target}'. Iterazione saltata.")
            continue

        print(f"Campioni della classe target nel test set: {len(X_test_target_df)}")

        # Addestro i modelli branch e siamese sui dati senza la classe target
        branch_model, siamese_model = train_branch_and_siamese(
            X_tv_filtrato, y_tv_filtrato,
            x_val_df.iloc[:, :genes_len], y_val_enc,
            genes_len, n_classes
        )

        branch_extractor = load_model('model/siamese_branch_temp.h5')
        branch_extractor.trainable = False

        # Preparazione dei riferimenti di tutte le classi presenti nel test set (tranne la classe target)
        references = {}
        all_classes_in_test = sorted(X_test_filtrato_df[label_column].unique())
        for cls in all_classes_in_test:
            idx_cls = X_test_filtrato_df[X_test_filtrato_df[label_column] == cls].index.tolist()

            # Prendo fino a 5 campioni per classe come riferimento
            if len(idx_cls) > 5:
                idx_cls = random.sample(idx_cls, 5)

            samples = []
            for idx in idx_cls:
                sample_idx = list(x_test_df.index).index(idx)
                samples.append(X_test_resh[sample_idx])

            # Calcolo gli embedding per i campioni di riferimento
            embeddings = [branch_extractor.predict(np.expand_dims(sample, axis=0), verbose=0)[0] for sample in samples]
            references[cls] = np.stack(embeddings, axis=0)

        # Valutazione di tutti i campioni della classe target
        predizioni = []
        y_true = []
        y_pred = []

        print(f"\nValutazione di {len(X_test_target_df)} campioni della classe '{classe_target}':")

        for idx in X_test_target_df.index:
            sample_idx = list(x_test_df.index).index(idx)
            sample = X_test_resh[sample_idx]

            # Calcolo l'embedding del campione target
            sample_embedding = branch_extractor.predict(np.expand_dims(sample, axis=0), verbose=0)[0]

            # Calcolo le distanze medie rispetto a tutte le classi di riferimento
            distanza_media = {}
            for cls, ref_embeddings in references.items():
                dists = np.linalg.norm(ref_embeddings - sample_embedding, axis=1)
                distanza_media[cls] = np.mean(dists)

            # Trovo la classe più vicina
            classe_predetta = min(distanza_media, key=distanza_media.get)

            predizioni.append({
                "Sample_ID": int(idx),
                "Classe_vera": classe_target,
                "Classe_predetta": classe_predetta,
                "E_Normal": classe_predetta.lower() == "normal"
            })

            y_true.append(classe_target)
            y_pred.append(classe_predetta)

        # Calcolo delle statistiche di classificazione
        classi_uniche = sorted(set(y_true + y_pred))
        conf_matrix = confusion_matrix(y_true, y_pred, labels=classi_uniche)
        conf_matrix_df = pd.DataFrame(conf_matrix, index=classi_uniche, columns=classi_uniche)

        print("\nMatrice di confusione:")
        print(conf_matrix_df)

        # Conteggio delle predizioni per ogni classe
        counts = {}
        for pred in y_pred:
            counts[pred] = counts.get(pred, 0) + 1

        # Calcolo delle percentuali per ogni classe
        class_distribution = []
        for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
            percentage = (count / len(y_pred)) * 100
            class_distribution.append({
                "classe": cls,
                "conteggio": count,
                "percentuale": round(percentage, 2)
            })
            print(f"{cls}: {count} campioni ({percentage:.2f}%)")

        # Calcolo della percentuale normal vs non-normal
        normal_count = sum(1 for pred in y_pred if pred.lower() == "normal")
        normal_percentage = (normal_count / len(y_pred)) * 100
        non_normal_percentage = 100 - normal_percentage

        print(f"\nPercentuale classificata come 'Normal': {normal_percentage:.2f}%")
        print(f"Percentuale classificata come NON 'Normal': {non_normal_percentage:.2f}%")

        # Aggiunta dei risultati
        risultati.append({
            "Classe_target": classe_target,
            "Totale_campioni": len(X_test_target_df),
            "Distribuzione": class_distribution,
            "Normal_percentage": round(normal_percentage, 2),
            "Non_normal_percentage": round(non_normal_percentage, 2)
        })

    # Creazione di un json
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_path = os.path.join(results_dir, "One_vs_n-1.json")
    with open(results_path, 'w') as f:
        json.dump(risultati, f, indent=4)
    print(f"\nI risultati sono stati salvati in '{results_path}'.")


if __name__ == "__main__":
    main()

