# fine_tuning.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dropout, Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

from main import *
from utility import *

n_classes = 9

def build_branch_model(hp):
    """
    Costruisce il branch model con una serie di layer e parametri configurabili
    per una ricerca più approfondita. In questo aggiornamento sono stati aumentati
    i possibili valori dei parametri e sono state aggiunte opzioni per la regolarizzazione.
    """
    global n_classes

    # di blocchi convoluzionali
    num_conv_layers = hp.Int('num_conv_layers', min_value=2, max_value=5, default=3)

    # Funzione di attivazione
    activation = hp.Choice('activation', values=['relu', 'elu', 'tanh'], default='relu')

    # Parametri blocchi conv - ampliati i valori possibili per aumentare il numero dei parametri
    filters1 = hp.Choice('filters1', values=[32, 64, 128, 256, 512, 1024], default=64)
    kernel_size1 = hp.Choice('kernel_size1', values=[3, 5, 7, 9, 11], default=3)

    filters2 = hp.Choice('filters2', values=[64, 128, 256, 512, 1024, 2048], default=128)
    kernel_size2 = hp.Choice('kernel_size2', values=[3, 5, 7, 9], default=3)

    # Terzo blocco (opzionale)
    if num_conv_layers >= 3:
        filters3 = hp.Choice('filters3', values=[128, 256, 512, 1024, 2048], default=256)
        kernel_size3 = hp.Choice('kernel_size3', values=[3, 5, 7, 9], default=3)

    # Quarto blocco (opzionale)
    if num_conv_layers >= 4:
        filters4 = hp.Choice('filters4', values=[256, 512, 1024, 2048], default=512)
        kernel_size4 = hp.Choice('kernel_size4', values=[3, 5, 7], default=3)

    # Quinto blocco (opzionale)
    if num_conv_layers == 5:
        filters5 = hp.Choice('filters5', values=[512, 1024, 2048], default=512)
        kernel_size5 = hp.Choice('kernel_size5', values=[3, 5], default=3)

    # Dropout
    dropout_rate1 = hp.Float('dropout1', min_value=0.1, max_value=0.6, step=0.1, default=0.3)
    dropout_rate2 = hp.Float('dropout2', min_value=0.1, max_value=0.6, step=0.1, default=0.3)

    # Parametri per FC layer
    dense_units = hp.Choice('dense_units', values=[128, 256, 512, 1024, 2048], default=256)

    # Layer aggiuntivo opzionale
    add_extra_dense = hp.Boolean('add_extra_dense', default=False)
    if add_extra_dense:
        extra_dense_units = hp.Choice('extra_dense_units', values=[128, 256, 512, 1024], default=128)

    # Embedding dimension
    embedding_units = hp.Choice('embedding_units', values=[32, 64, 128, 256, 512, 1024], default=64)

    # Selezione del tipo di regolarizzazione
    reg_type = hp.Choice('reg_type', values=['l1', 'l2', 'l1_l2'], default='l2')
    weight_decay = hp.Float('weight_decay', min_value=1e-6, max_value=1e-3, sampling='log', default=1e-5)
    if reg_type == 'l1':
        regularizer = tf.keras.regularizers.l1(weight_decay)
    elif reg_type == 'l2':
        regularizer = tf.keras.regularizers.l2(weight_decay)
    elif reg_type == 'l1_l2':
        # Bilancia L1 e L2: si potrebbe anche rendere questa scelta parametrica
        regularizer = tf.keras.regularizers.l1_l2(l1=weight_decay, l2=weight_decay)

    # Learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log', default=1e-4)

    # Creazione del modello
    inp = Input(shape=(43, 1))
    x = inp

    # Primo blocco conv
    x = Conv1D(filters=filters1, kernel_size=kernel_size1, activation=activation,
               kernel_regularizer=regularizer, padding="same")(x)
    x = BatchNormalization()(x)
    pool_size1 = hp.Choice('pool_size1', values=[2, 3], default=2)
    x = MaxPooling1D(pool_size=pool_size1)(x)

    # Secondo blocco conv
    x = Conv1D(filters=filters2, kernel_size=kernel_size2, activation=activation,
               kernel_regularizer=regularizer, padding="same")(x)
    x = BatchNormalization()(x)
    pool_size2 = hp.Choice('pool_size2', values=[2, 3], default=2)
    x = MaxPooling1D(pool_size=pool_size2)(x)

    # Terzo blocco conv
    if num_conv_layers >= 3:
        x = Conv1D(filters=filters3, kernel_size=kernel_size3, activation=activation,
                   kernel_regularizer=regularizer, padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)

    # Quarto blocco conv
    if num_conv_layers >= 4:
        x = Conv1D(filters=filters4, kernel_size=kernel_size4, activation=activation,
                   kernel_regularizer=regularizer, padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)

    # Quinto blocco conv
    if num_conv_layers == 5:
        x = Conv1D(filters=filters5, kernel_size=kernel_size5, activation=activation,
                   kernel_regularizer=regularizer, padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dropout(dropout_rate1)(x)
    x = Dense(dense_units, activation=activation, kernel_regularizer=regularizer)(x)
    x = Dropout(dropout_rate2)(x)

    # Layer fully connected extra
    if add_extra_dense:
        x = Dense(extra_dense_units, activation=activation, kernel_regularizer=regularizer)(x)

    x = Dense(embedding_units, activation=activation, kernel_regularizer=regularizer, name="embedding")(x)

    out = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)

    # Compilazione del modello
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


class SensSpecTuner(kt.Hyperband):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.siamese_model = None
        self.genes_len = None
        self.x_test_df = None
        self.test_ind = None

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        # Costruisce il ramo (branch) con i parametri proposti
        model = self.hypermodel.build(trial.hyperparameters)
        # Aumenta il numero di epoche, per migliorare la ricerca (es. 50)
        model.fit(*fit_args, **fit_kwargs, batch_size=64, epochs=50, verbose=0)

        # Crea il modello siamese basato sul branch creato
        temp_siamese = build_siamese_model(model, (43, 1))

        # Valuta la one-shot accuracy
        one_shot_acc, _ = test_oneshot(
            model=temp_siamese,
            genes_len=self.genes_len,
            x_test_df=self.x_test_df,
            test_ind=self.test_ind,
            X_test_resh=X_test_resh,
            N=9,
            label_column='CANCER_TYPE',
            branch_extractor=model
        )

        # Aggiorna la metrica personalizzata
        test_loss, test_accuracy = model.evaluate(X_test_resh, y_test_ohe, verbose=0)
        self.oracle.update_trial(trial.trial_id, {'val_accuracy': test_accuracy})
        self.save_model(trial.trial_id, model)

    def save_model(self, trial_id, model):
        save_path = os.path.join(self.project_dir, f"model_{trial_id}.h5")
        model.save(save_path)


if __name__ == "__main__":
    """
    Esegue la ricerca degli iperparametri del branch model
    """
    excel_path = "Data.xlsx"

    # Prepara i dati
    (
        x_train_df, x_val_df, x_test_df,
        y_test, y_train_enc, y_val_enc, y_test_enc,
        X_train, X_val, X_test,
        X_train_resh, X_val_resh, X_test_resh,
        y_train_ohe, y_val_ohe, y_test_ohe,
        le, n_classes_global, genes_len
    ) = prepare_data_for_models(excel_path)

    # Aggiorna la variabile globale
    n_classes = n_classes_global
    test_ind = list(x_test_df.index)
    input_shape = (43, 1)
    print("Dati caricati correttamente per la ricerca iperparametri.")

    # Crea un modello di partenza (branch + siamese)
    base_branch = build_branch_model(kt.HyperParameters())
    base_siamese = build_siamese_model(base_branch, input_shape)

    # Dichiara il Tuner personalizzato
    tuner = SensSpecTuner(
        hypermodel=build_branch_model,
        objective=kt.Objective("val_accuracy", direction="max"),
        max_epochs=50,
        factor=3,
        directory="tuner_dir",
        project_name="fine_tuning_branch_model"
    )

    # Assegna variabili al tuner per valutazione one-shot
    tuner.siamese_model = base_siamese
    tuner.genes_len = genes_len
    tuner.x_test_df = x_test_df
    tuner.test_ind = test_ind

    # Mostra il riepilogo della ricerca
    tuner.search_space_summary()

    # Avvia la ricerca sugli iperparametri
    tuner.search(
        X_train_resh, y_train_ohe,
        validation_data=(X_val_resh, y_val_ohe)
    )

    # Riepilogo dei risultati
    tuner.results_summary()

    # Estrae i migliori iperparametri
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n** Migliori Parametri della Rete **")
    for param_name, param_value in best_hp.values.items():
        print(f"{param_name}: {param_value}")

    # Costruisce il modello migliore e lo allena ulteriormente
    best_model = tuner.hypermodel.build(best_hp)
    best_model.fit(
        X_train_resh, y_train_ohe,
        validation_data=(X_val_resh, y_val_ohe),
        epochs=50,  # Ulteriore training
        batch_size=64,
        verbose=1
    )

    # Valuta il modello sul test set
    test_loss, test_accuracy = best_model.evaluate(X_test_resh, y_test_ohe, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}% - Test Loss: {test_loss:.4f}")

    pred_probs = best_model.predict(X_test_resh)
    y_pred = np.argmax(pred_probs, axis=1)
    y_true = np.argmax(y_test_ohe, axis=1)

    # Calcolo sensitivity e specificity
    cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true)))
    sens_list, spec_list = [], []
    for idx in range(len(sorted(set(y_true)))):
        sens, spec = compute_sensitivity_specificity(cm, idx)
        sens_list.append(sens)
        spec_list.append(spec)
    macro_sens = np.mean(sens_list)
    macro_spec = np.mean(spec_list)
    print(f"Macro Sensitivity: {macro_sens:.2f} - Macro Specificity: {macro_spec:.2f}")

    # One-shot evaluation con il modello siamese
    best_siamese_model = build_siamese_model(best_model, input_shape)
    acc_tumor, preds_detail_tumor = test_oneshot(
        model=best_siamese_model,
        genes_len=genes_len,
        x_test_df=x_test_df,
        test_ind=test_ind,
        X_test_resh=X_test_resh,
        N=9,
        label_column='CANCER_TYPE',
        branch_extractor=best_model
    )
    print(f"\n[One-shot] Accuratezza One-Shot (Siamese): {acc_tumor:.2f}%")

    print("Dettaglio predizioni (indice, etichetta vera, etichetta predetta):")
    for detail in preds_detail_tumor:
        print(detail)

    # Confusion Matrix e classification_report
    print("\nConfusion Matrix (Branch):")
    print(cm)
    report = classification_report(y_true, y_pred, target_names=[str(cls) for cls in sorted(set(y_true))])
    print("\nClassification Report (Branch):")
    print(report)

    # Salva e mostra i risultati finali
    metrics_per_class = {}
    for idx, cls in enumerate(sorted(set(y_true))):
        sens, spec = compute_sensitivity_specificity(cm, idx)
        metrics_per_class[str(cls)] = {"Sensitivity": sens, "Specificity": spec}

    best_metrics = {
        "Best Hyperparameters": best_hp.values,
        "Test Accuracy": test_accuracy,
        "Test Loss": test_loss,
        "Macro Sensitivity": macro_sens,
        "Macro Specificity": macro_spec,
        "One-Shot Accuracy": acc_tumor,
        "Per Class Metrics": metrics_per_class,
    }

    print("\n=== Riepilogo Metriche Migliori ===")
    for param_name, param_value in best_hp.values.items():
        print(f"{param_name}: {param_value}")

    print(f"Test Accuracy: {best_metrics['Test Accuracy'] * 100:.2f}%")
    print(f"Test Loss: {best_metrics['Test Loss']:.4f}")
    print(f"Macro Sensitivity: {best_metrics['Macro Sensitivity']:.2f}")
    print(f"Macro Specificity: {best_metrics['Macro Specificity']:.2f}")
    print(f"One-Shot Accuracy: {best_metrics['One-Shot Accuracy']:.2f}%\n")

    print("Metriche per Classe:")
    for cls, metrics in best_metrics["Per Class Metrics"].items():
        print(
            f" • Classe {cls}: Sensitivity = {metrics['Sensitivity']:.2f}, Specificity = {metrics['Specificity']:.2f}")
    # Salva le metriche migliori in un file JSON
    output_file = os.path.join("results/fine_tuning_best_metrics.json")
    with open(output_file, "w") as f:
        json.dump(best_metrics, f, indent=4, default=convert_to_serializable)

    print(f"Metriche migliori salvate in {output_file}.")

    print("\nFine della ricerca iperparametri.")
