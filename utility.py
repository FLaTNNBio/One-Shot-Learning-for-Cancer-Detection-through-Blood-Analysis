# utility.py

import tensorflow.keras.backend as K
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, Flatten, MaxPooling1D, Lambda, BatchNormalization, Activation)
from tensorflow.keras.models import Model
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import random


def initialize_bias(shape, name=None, dtype=None):
    """
    Inizializza i bias con una distribuzione normale centrata su 0.5.
    """
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def last_layer(encoded_l, encoded_r, distance='L2'):
    """
    Crea l'ultimo layer della rete Siamese in base al tipo di distanza.
    """
    if distance == 'L1':
        L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])
        dense_1 = Dense(512, activation='relu', bias_initializer=initialize_bias)(L1_distance)
        prediction = Dense(1, activation='sigmoid')(dense_1)
        return prediction

    elif distance == 'L2':
        # L2 distance (normalizzata)
        L2_layer = Lambda(lambda tensors: (tensors[0] - tensors[1]) ** 2 /
                                          (tensors[0] + tensors[1] + K.epsilon()))
        L2_distance = L2_layer([encoded_l, encoded_r])
        dense_1 = Dense(256, activation='relu', bias_initializer=initialize_bias)(L2_distance)
        prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(dense_1)
        return prediction

    else:
        # 'cosine' distance
        cos_layer = Lambda(lambda tensors:
                           K.sum(tensors[0] * tensors[1], axis=-1, keepdims=True) /
                           (K.sqrt(K.sum(tensors[0] ** 2, axis=-1, keepdims=True) + K.epsilon()) *
                            K.sqrt(K.sum(tensors[1] ** 2, axis=-1, keepdims=True) + K.epsilon())))
        cos_distance = cos_layer([encoded_l, encoded_r])
        # Più alto = più simile => output ~ [0..1]
        prediction = Activation('sigmoid')(cos_distance)
        return prediction


def compute_macro_metrics(y_true, y_pred):
    """
    Calcola la sensitivity e la specificity medie basandosi sulla confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    sens_list, spec_list = [], []
    for idx in range(cm.shape[0]):
        TP = cm[idx, idx]
        FN = np.sum(cm[idx, :]) - TP
        FP = np.sum(cm[:, idx]) - TP
        TN = np.sum(cm) - (TP + FN + FP)
        sens = TP / (TP + FN) if (TP + FN) != 0 else 0
        spec = TN / (TN + FP) if (TN + FP) != 0 else 0
        sens_list.append(sens)
        spec_list.append(spec)
    macro_sens = np.mean(sens_list)
    macro_spec = np.mean(spec_list)
    return macro_sens, macro_spec


def print_confusion_matrix(cm, class_labels):
    # Stampa header
    header = "Confusion Matrix:\n\t" + "\t".join(str(lbl) for lbl in class_labels)
    print(header)
    # Stampa ogni riga formattata con il nome della classe a sinistra
    for i, row in enumerate(cm):
        row_str = "\t".join(f"{val:3d}" for val in row)
        print(f"{class_labels[i]}\t{row_str}")


def test_oneshot(model, genes_len, x_test_df, test_ind, X_test_resh, N=9, label_column='Tumor_Type', branch_extractor=None):
    """
    Esegue il test one-shot: per ogni campione di test si estrae l'embedding
    e lo confronta con un embedding di riferimento per ciascuna classe.
    """
    # Verifica che la colonna esista
    if label_column not in x_test_df.columns:
        raise KeyError(f"La colonna '{label_column}' non è presente nel DataFrame dei dati di test. "
                       "Verifica il nome della colonna delle etichette.")

    # Se non abbiamo passato branch_extractor, provo a costruirlo dal modello
    if branch_extractor is None:
        try:
            branch_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
        except Exception as e:
            raise ValueError("Impossibile creare il feature extractor. "
                             "Per modelli con più input (come il modello Siamese) "
                             "passare il parametro branch_extractor.") from e

    # Otteniamo le classi uniche dalla colonna specificata
    unique_classes = sorted(x_test_df[label_column].unique())
    references = {}

    # Per ogni classe, scegliamo un campione casuale dal test set
    for cls in unique_classes:
        indices_cls = x_test_df.index[x_test_df[label_column] == cls].tolist()
        ref_idx = random.choice(indices_cls)
        # Recupero embedding dal branch extractor
        ref_embedding = branch_extractor.predict(np.expand_dims(X_test_resh[ref_idx], axis=0))[0]
        references[cls] = ref_embedding

    predictions = []
    for i in test_ind:
        test_embedding = branch_extractor.predict(np.expand_dims(X_test_resh[i], axis=0))[0]
        # Calcola la distanza L2 tra embedding del campione e ciascun embedding di riferimento
        distances = {cls: np.linalg.norm(test_embedding - emb) for cls, emb in references.items()}
        pred_class = min(distances, key=distances.get)
        predictions.append(pred_class)

    # Estrazione delle etichette vere
    true_labels = x_test_df.loc[test_ind, label_column].tolist()
    acc = np.mean([pred == true for pred, true in zip(predictions, true_labels)]) * 100
    preds_detail = list(zip(test_ind, true_labels, predictions))
    return acc, preds_detail


def test_kshot(model, genes_len, x_test_df, test_ind, X_test_resh, k_shot=5,
               label_column='CANCER_TYPE', branch_extractor=None):
    """
    Esegue il test k-shot escludendo il campione query dal gruppo di riferimento se appartiene alla stessa classe.
    Se k_shot è impostato a "all", vengono usati tutti i campioni disponibili per la classe.
    """
    if branch_extractor is None:
        try:
            branch_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
        except Exception as e:
            raise ValueError(
                "Impossibile creare il feature extractor. Per modelli con più input (come il modello Siamese) passare il parametro branch_extractor.") from e

    unique_classes = sorted(x_test_df[label_column].unique())
    # Pre-calcola i riferimenti per ogni classe
    references = {}
    for cls in unique_classes:
        indices_cls = list(x_test_df.index[x_test_df[label_column] == cls])
        if isinstance(k_shot, str) and k_shot.lower() == "all":
            selected_indices = indices_cls
        else:
            selected_indices = random.sample(indices_cls, min(k_shot, len(indices_cls)))
        embeddings = [branch_extractor.predict(np.expand_dims(X_test_resh[idx], axis=0))[0]
                      for idx in selected_indices]
        references[cls] = np.stack(embeddings, axis=0)

    predictions = []
    for i in test_ind:
        current_class = x_test_df.loc[i, label_column]
        sample_embedding = branch_extractor.predict(np.expand_dims(X_test_resh[i], axis=0))[0]

        # Escludo il campione corrente se presente nel riferimento
        indices_cls = list(x_test_df.index[x_test_df[label_column] == current_class])
        if i in indices_cls:
            indices_cls.remove(i)
        if len(indices_cls) > 0:
            if isinstance(k_shot, str) and k_shot.lower() == "all":
                selected_indices = indices_cls
            else:
                selected_indices = random.sample(indices_cls, min(k_shot, len(indices_cls)))
            embeddings = [branch_extractor.predict(np.expand_dims(X_test_resh[idx], axis=0))[0]
                          for idx in selected_indices]
            ref_embedding = np.mean(embeddings, axis=0)
        else:
            ref_embedding = np.mean(references[current_class], axis=0)

        distances = {}
        # Calcola la distanza media tra il campione corrente e i riferimenti di ciascuna classe
        for cls, ref_embeddings in references.items():
            if cls == current_class and len(indices_cls) > 0:
                distances[cls] = np.linalg.norm(ref_embedding - sample_embedding)
            else:
                dists = np.linalg.norm(ref_embeddings - sample_embedding, axis=1)
                distances[cls] = np.mean(dists)
        pred_class = min(distances, key=distances.get)
        predictions.append(pred_class)

    true_labels = list(x_test_df.loc[test_ind, label_column])
    acc = np.mean([pred == true for pred, true in zip(predictions, true_labels)]) * 100
    preds_detail = list(zip(test_ind, true_labels, predictions))
    return acc, preds_detail


def compute_confusion_metrics(ref_class, preds_detail, label_column='CANCER_TYPE'):
    """
    Calcola la matrice di confusione in formato binario per la classe ref_class.
    La classificazione è considerata positiva se la previsione (o l'etichetta vera) corrisponde a ref_class,
    altrimenti negativa.
    """
    TP = FP = FN = TN = 0
    for idx, true, pred in preds_detail:
        # Se il campione appartiene alla classe rimossa
        if true == ref_class:
            if pred == ref_class:
                TP += 1
            else:
                FN += 1
        else:
            if pred == ref_class:
                FP += 1
            else:
                TN += 1

    # Calcola i valori; attenzione a evitare divisioni per 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # recall
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    return TP, FN, FP, TN, accuracy, sensitivity, specificity


def create_siamese_pairs(X, y):
    """
    Crea coppie di input per il training del modello Siamese.
    """
    pairs_A, pairs_B, labels = [], [], []

    # Crea un dizionario di indici per ciascuna classe
    indices_per_class = {}
    for idx, label in enumerate(y):
        indices_per_class.setdefault(label, []).append(idx)

    unique_labels = list(indices_per_class.keys())

    # Genera coppie per ogni classe
    for label in unique_labels:
        indices = indices_per_class[label]
        n = len(indices)
        # Coppie positive
        for i in range(n - 1):
            idx1, idx2 = indices[i], indices[i + 1]
            pairs_A.append(X[idx1])
            pairs_B.append(X[idx2])
            labels.append(1)
            # Coppia negativa: prendi un indice da un'altra classe
            neg_label = random.choice([l for l in unique_labels if l != label])
            neg_idx = random.choice(indices_per_class[neg_label])
            pairs_A.append(X[idx1])
            pairs_B.append(X[neg_idx])
            labels.append(0)

    return np.array(pairs_A), np.array(pairs_B), np.array(labels)


def plot_training(history, filename):
    """
    Genera e salva i grafici dell'andamento di accuracy e loss
    durante il training e la validation.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # Crea la figura
    plt.figure(figsize=(12, 5))

    # Subplot per l'accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'orange', label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Subplot per la loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'orange', label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Salva la figura nella directory corrente
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_and_save_confusion_matrix(y_true, y_pred, classes, filename="confusion_matrix.png"):
    """
    Genera e salva la confusion matrix come immagine.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(10, 8))

    # Creazione della heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Salvataggio
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix salvata come {filename}")


def compute_additional_metrics(y_true, y_pred, classes):
    """
    Calcola precision, recall, F1-score e restituisce un report text-based.
    """
    report = classification_report(y_true, y_pred, labels=classes, target_names=classes, zero_division=0)
    print("\nClassification Report:")
    print(report)
    return report


def plot_and_save_confusion_matrix(y_true, y_pred, classes, filename="confusion_matrix.png"):
    """
    Genera e salva la confusion matrix come immagine.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Etichette Predette')
    plt.ylabel('Etichette Vere')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Confusion Matrix salvata in {filename}")


def compute_additional_metrics(y_true, y_pred, classes):
    """
    Calcola Precision, Recall, F1-score e ritorna un report testuale.
    """
    report = classification_report(y_true, y_pred, labels=classes, target_names=classes, zero_division=0)
    print("\nClassification Report:")
    print(report)
    return report


def plot_and_save_confusion_matrix(y_true, y_pred, classes, directory=".", filename="confusion_matrix_branch.png"):
    """
    Genera e salva la confusion matrix nella directory specificata.
    """
    # Se le etichette sono indici interi, convertili in nome usando la lista delle classi
    if isinstance(y_true[0], int):
        y_true_named = [classes[i] for i in y_true]
    else:
        y_true_named = y_true

    if isinstance(y_pred[0], int):
        y_pred_named = [classes[i] for i in y_pred]
    else:
        y_pred_named = y_pred

    # Utilizza pandas crosstab per creare la confusion matrix includendo tutte le classi
    cm_df = pd.crosstab(
        pd.Series(y_true_named, name="Vero"),
        pd.Series(y_pred_named, name="Predetto")
    ).reindex(index=classes, columns=classes, fill_value=0)

    # Crea il grafico della heatmap usando il DataFrame
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Etichette Predette')
    plt.ylabel('Etichette Vere')
    plt.title('Confusion Matrix (Siamese Model)')
    plt.tight_layout()

    # Se la directory non esiste, creala
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Percorso completo del file
    filepath = os.path.join(directory, filename)

    # Salva l'immagine e chiudi il plot
    plt.savefig(filepath)
    plt.close()
    print(f"Confusion Matrix salvata come immagine in {filepath}.")


def last_layer(encoded_l, encoded_r, distance='L2'):
    """
    Crea l'ultimo layer della rete Siamese in base al tipo di distanza.
    """
    if distance == 'L1':
        L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])
        dense_1 = Dense(512, activation='relu', bias_initializer=initialize_bias)(L1_distance)
        prediction = Dense(1, activation='sigmoid')(dense_1)
        return prediction

    elif distance == 'L2':
        # L2 distance (normalizzata)
        L2_layer = Lambda(lambda tensors: (tensors[0] - tensors[1]) ** 2 /
                                          (tensors[0] + tensors[1] + K.epsilon()))
        L2_distance = L2_layer([encoded_l, encoded_r])
        dense_1 = Dense(256, activation='relu', bias_initializer=initialize_bias)(L2_distance)
        prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(dense_1)
        return prediction

    else:
        # 'cosine' distance
        cos_layer = Lambda(lambda tensors:
                           K.sum(tensors[0] * tensors[1], axis=-1, keepdims=True) /
                           (K.sqrt(K.sum(tensors[0] ** 2, axis=-1, keepdims=True) + K.epsilon()) *
                            K.sqrt(K.sum(tensors[1] ** 2, axis=-1, keepdims=True) + K.epsilon())))
        cos_distance = cos_layer([encoded_l, encoded_r])
        # Più alto = più simile => output ~ [0..1]
        prediction = Activation('sigmoid')(cos_distance)
        return prediction


def extended_classification_report(y_true, y_pred, target_names):
    """
    Calcola un report di classificazione esteso che include precision, sensitivity (recall),
    specificity, f1-score e support per ciascuna classe.
    """
    # Assicuriamoci che le etichette siano tutte stringhe
    y_true_str = [str(label) for label in y_true]
    y_pred_str = [str(label) for label in y_pred]

    # Calcola precision, recall, f1-score e support usando sklearn utilizzando i nomi delle classi
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_str, y_pred_str, labels=target_names, zero_division=0
    )

    # Calcola la matrice di confusione usando i nomi delle classi
    cm = confusion_matrix(y_true_str, y_pred_str, labels=target_names)
    specificity = []

    # Calcola la specificità per ogni classe
    for i in range(len(target_names)):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FN + FP)
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        specificity.append(spec)

    # Crea un DataFrame per organizzare i risultati in maniera tabellare
    report_df = pd.DataFrame({
        'Classe': target_names,
        'Precision': precision,
        'Sensitivity (Recall)': recall,  # recall equivale a sensitivity
        'Specificity': specificity,
        'F1-score': f1,
        'Support': support
    })

    # Imposta la colonna "Classe" come indice del DataFrame
    report_df.set_index('Classe', inplace=True)

    # Format del report come stringa
    report_str = report_df.to_string(float_format=lambda x: f"{x:0.2f}")
    return report_str


def convert_to_serializable(obj):
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return str(obj)


def compute_sensitivity_specificity(cm, class_index):
    TP = cm[class_index, class_index]
    FN = np.sum(cm[class_index, :]) - TP
    FP = np.sum(cm[:, class_index]) - TP
    TN = np.sum(cm) - (TP + FN + FP)
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    return sensitivity, specificity


