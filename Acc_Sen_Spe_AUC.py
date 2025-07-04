# Acc_Sen_Spe_AUC.py

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import pandas as pd
import numpy as np
import json
from preprocessing import prepare_data_for_models
from utility import *

def calcola_metriche_da_lista(predictions: list):
    df = pd.DataFrame(predictions)
    df["predicted_label_lower"] = df["predicted_label"].str.lower()
    df["true_lower"] = df["true"].str.lower()

    # Tabella distribuzione sample per classe e per stadio
    if not df['true'].apply(lambda x: isinstance(x, str)).all():
        df['true'] = df['true'].apply(lambda x: x[0] if hasattr(x, "__getitem__") and not isinstance(x, str) else x)

    if "AJCC_Stage" in df.columns:
        tabella = pd.pivot_table(
            df,
            values='predicted_label',
            index='true',
            columns='AJCC_Stage',
            aggfunc='count',
            fill_value=0
        ).rename_axis("Classe").rename_axis("AJCC_Stage", axis=1)
        tabella["Totale"] = tabella.sum(axis=1)
        print("\nTabella di conteggio sample per classe e per stadio AJCC:")
        print("----------------------------------------------------------")
        print(tabella)
        print()
    else:
        print("Attenzione: la colonna 'AJCC_Stage' non è presente, tabella non creata.")

    # Calcolo dei predetti "Normal" vs "Non Normal"
    count_normal = df[df["predicted_label_lower"] == "normal"].shape[0]
    count_non_normal = df[df["predicted_label_lower"] != "normal"].shape[0]

    print("Risultati globali delle predizioni:")
    print("------------------------------------")
    print(f"Totale predizioni     : {df.shape[0]}")
    print(f"Predizioni 'Normal'    : {count_normal}")
    print(f"Predizioni 'Non Normal': {count_non_normal}\n")

    # Dettaglio per le predizioni "Normal"
    normal_df = df[df["predicted_label_lower"] == "normal"]
    if not normal_df.empty:
        normal_correct = normal_df[normal_df["true_lower"] == "normal"].shape[0]
        normal_incorrect = normal_df[normal_df["true_lower"] != "normal"].shape[0]
        print("Dettaglio per le predizioni 'Normal':")
        print("-------------------------------------")
        print(f"Predizioni 'Normal' corrette (true == 'normal'): {normal_correct}")
        print(f"Predizioni 'Normal' errate  (true != 'normal'): {normal_incorrect}\n")

        # Breakdown per AJCC Stage per i record "Normal" errati
        normal_errors_df = normal_df[normal_df["true_lower"] != "normal"]
        if "AJCC_Stage" in df.columns:
            normal_errors_df = normal_errors_df[normal_errors_df["AJCC_Stage"].notna()]
            if not normal_errors_df.empty:
                print("Breakdown per stadio per le predizioni 'Normal' errate (true != 'normal'):")
                print("------------------------------------------------------------------------")
                stage_counts = normal_errors_df["AJCC_Stage"].value_counts().sort_index()
                for stage, count in stage_counts.items():
                    print(f"Stadio {stage}: {count}")
                    stage_details = normal_errors_df[normal_errors_df["AJCC_Stage"] == stage][
                        ["true", "predicted_label", "AJCC_Stage"]
                    ]
                    print(stage_details.to_string(index=False))
                    print()  # Riga vuota di separazione
            else:
                print("Non sono presenti errori per le predizioni 'Normal' (true != 'normal').\n")
        else:
            print("Attenzione: La chiave 'AJCC_Stage' non è presente, impossibile calcolare il breakdown per gli errori 'Normal'.\n")


    # Dettaglio per le predizioni "Non Normal"
    non_normal_df = df[df["predicted_label_lower"] != "normal"]
    if not non_normal_df.empty:
        non_normal_correct = non_normal_df[non_normal_df["true_lower"] != "normal"].shape[0]
        non_normal_incorrect = non_normal_df[non_normal_df["true_lower"] == "normal"].shape[0]
        print("Dettaglio per le predizioni 'Non Normal':")
        print("-----------------------------------------")
        print(f"Predizioni 'Non Normal' corrette (true != 'normal'): {non_normal_correct}")
        print(f"Predizioni 'Non Normal' errate  (true == 'normal'): {non_normal_incorrect}\n")

    # Breakdown per AJCC Stage per le predizioni "Non Normal"
    if "AJCC_Stage" in df.columns:
        non_normal_stage_df = non_normal_df[non_normal_df["AJCC_Stage"].notna()]
        stage_counts = non_normal_stage_df["AJCC_Stage"].value_counts().sort_index()

        print("Breakdown per stadio per le predizioni non normali:")
        print("----------------------------------------------------")
        for stage, count in stage_counts.items():
            print(f"Stadio {stage}: {count}")
        print()  # Riga vuota di separazione

        # Dettaglio per ciascuno stadio specifico (0, 1, 2 e 3) per i record "Non Normal"
        for stage in [0, 1, 2, 3]:
            stage_df = df[(df["AJCC_Stage"] == stage) & (df["predicted_label_lower"] != "normal")]
            if not stage_df.empty:
                print(f"Dettagli per i casi con AJCC_Stage pari a {stage} e classificati come 'Non Normal':")
                print("-------------------------------------------------------------------------")
                print(stage_df[["true", "predicted_label", "AJCC_Stage"]].to_string(index=False))
                print()  # Riga vuota di separazione
            else:
                print(f"Non sono presenti record con AJCC_Stage pari a {stage} e classificati come 'Non Normal'.\n")

        # Breakdown degli errori per i record "Non Normal" (true == 'normal')
        errors_df = df[(df["predicted_label_lower"] != "normal") & (df["true_lower"] == "normal")]
        if not errors_df.empty:
            stage_error_counts = errors_df["AJCC_Stage"].value_counts().sort_index()
            print("Breakdown degli errori per stadio per le predizioni 'Non Normal' errate (true == 'normal'):")
            print("-------------------------------------------------------------------------------")
            for stage, count in stage_error_counts.items():
                print(f"Stadio {stage}: {count}")
            print()  # Riga vuota di separazione
        else:
            print("Non sono presenti errori nelle predizioni 'Non Normal' (true == 'normal').\n")
    else:
        print("Attenzione: La chiave 'AJCC_Stage' non è presente, impossibile calcolare il breakdown per stadio.")

    # Confusion Matrix binaria: Normal vs Non Normal
    TN_bin = df[(df["true_lower"] == "normal") & (df["predicted_label_lower"] == "normal")].shape[0]
    FP_bin = df[(df["true_lower"] == "normal") & (df["predicted_label_lower"] != "normal")].shape[0]
    FN_bin = df[(df["true_lower"] != "normal") & (df["predicted_label_lower"] == "normal")].shape[0]
    TP_bin = df[(df["true_lower"] != "normal") & (df["predicted_label_lower"] != "normal")].shape[0]

    sensitivity_bin = TP_bin / (TP_bin + FN_bin) if (TP_bin + FN_bin) > 0 else 0
    specificity_bin = TN_bin / (TN_bin + FP_bin) if (TN_bin + FP_bin) > 0 else 0

    mat = pd.DataFrame({
        "Predetto Normal (F)": [TN_bin, FN_bin],
        "Predetto Non Normal (T)": [FP_bin, TP_bin]
    }, index=["True Normal (F)", "True Non Normal (T)"])

    print("Confusion Matrix Binaria (Normal vs Non Normal):")
    print("-----------------------------------------------")
    print(mat.to_string(justify="center"))
    print()
    print(f"Sensitivity (TP / (TP + FN)): {sensitivity_bin:.4f} (TP={TP_bin}, FN={FN_bin})")
    print(f"Specificity (TN / (TN + FP)): {specificity_bin:.4f} (TN={TN_bin}, FP={FP_bin})\n")

    # Ulteriori metriche per AJCC Stage
    if "AJCC_Stage" in df.columns:
        stage_list = sorted([stage for stage in df["AJCC_Stage"].dropna().unique() if stage != 0])
        agg_list = []
        for stage in stage_list:
            stage_df = df[df["AJCC_Stage"] == stage]
            tp_s = stage_df[(stage_df["predicted_label_lower"] != "normal") & (stage_df["true_lower"] != "normal")].shape[0]
            fn_s = stage_df[(stage_df["predicted_label_lower"] == "normal") & (stage_df["true_lower"] != "normal")].shape[0]
            tn_s = stage_df[(stage_df["predicted_label_lower"] == "normal") & (stage_df["true_lower"] == "normal")].shape[0]
            fp_s = stage_df[(stage_df["predicted_label_lower"] != "normal") & (stage_df["true_lower"] == "normal")].shape[0]

            sensitivity_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0
            specificity_s = tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0

            agg_list.append({
                "AJCC_Stage": stage,
                "Sensitivity": sensitivity_s,
                "Specificity": specificity_s,
                "TP": tp_s,
                "FN": fn_s,
                "TN": tn_s,
                "FP": fp_s
            })

        agg_df = pd.DataFrame(agg_list).sort_values("AJCC_Stage")
        print("Confusion Matrix aggregata per AJCC Stage:")
        print("------------------------------------------")
        print(agg_df.to_string(index=False))
    else:
        print("Attenzione: 'AJCC_Stage' non è presente, quindi non si può calcolare la matrice aggregata per stadio.")
    # Tabella: Sensitivity e Specificity per tipo di cancro per ogni AJCC Stage
    # Tabella: Sensitivity e Specificity per tipo di cancro per ogni AJCC Stage (escludendo lo stadio 0)
    if "AJCC_Stage" in df.columns:
        # Consideriamo solo le classi di tumore (escludendo 'normal')
        cancer_classes = sorted([cls for cls in df["true_lower"].dropna().unique() if cls != "normal"])

        # Estrai solo gli stadi diversi da 0 e ordina in ordine crescente
        stages = sorted([stage for stage in df["AJCC_Stage"].dropna().unique() if stage != 0])

        results = []

        for stage in stages:
            stage_df = df[df["AJCC_Stage"] == stage]
            for cls in cancer_classes:
                # Calcolo: One-vs-Rest per la classe 'cls'
                TP = stage_df[(stage_df["true_lower"] == cls) & (stage_df["predicted_label_lower"] == cls)].shape[0]
                FN = stage_df[(stage_df["true_lower"] == cls) & (stage_df["predicted_label_lower"] != cls)].shape[0]
                FP = stage_df[(stage_df["true_lower"] != cls) & (stage_df["predicted_label_lower"] == cls)].shape[0]
                TN = stage_df.shape[0] - TP - FN - FP

                sensitivity = TP / (TP + FN) if (TP + FN) > 0 else None
                specificity = TN / (TN + FP) if (TN + FP) > 0 else None

                results.append({
                    "AJCC_Stage": stage,
                    "Classe": cls,
                    "Sensitivity": round(sensitivity, 4) if sensitivity is not None else "N/A",
                    "Specificity": round(specificity, 4) if specificity is not None else "N/A"
                })

        table_metrics = pd.DataFrame(results).sort_values(["AJCC_Stage", "Classe"])
        print("\nTabella: Sensitivity e Specificity per tipo di cancro per ogni AJCC Stage (stadio 0 escluso):")
        print("-----------------------------------------------------------------------------------------------")
        print(table_metrics.to_string(index=False))
    else:
        print("Attenzione: 'AJCC_Stage' non è presente, impossibile calcolare la tabella per stadio e tipo di cancro.")

    # Metriche per classe della variabile y in forma tabellare
    print("\nMetriche per classe della variabile y:")
    print("---------------------------------------------------")
    metrics_cls = []
    classi = sorted([cls for cls in df["true_lower"].dropna().unique() if cls != "normal"])
    totale = df.shape[0]
    for cls in classi:
        tp_cls = df[(df["true_lower"] == cls) & (df["predicted_label_lower"] == cls)].shape[0]
        fn_cls = df[(df["true_lower"] == cls) & (df["predicted_label_lower"] != cls)].shape[0]
        fp_cls = df[(df["true_lower"] != cls) & (df["predicted_label_lower"] == cls)].shape[0]
        tn_cls = totale - tp_cls - fn_cls - fp_cls

        sensitivity_cls = tp_cls / (tp_cls + fn_cls) if (tp_cls + fn_cls) > 0 else 0
        specificity_cls = tn_cls / (tn_cls + fp_cls) if (tn_cls + fp_cls) > 0 else 0

        metrics_cls.append({
            "Classe": cls,
            "Sensitivity": round(sensitivity_cls, 4),
            "Specificity": round(specificity_cls, 4)
        })

    metrics_df = pd.DataFrame(metrics_cls)
    print(metrics_df.to_string(index=False))

    # Aggiunta: Matrici di Confusione per ciascuna classe per ogni AJCC Stage
    if "AJCC_Stage" in df.columns:
        stages = sorted(df["AJCC_Stage"].dropna().unique())
        tutte_classi = sorted(set(df["true_lower"].dropna().unique()).union(set(df["predicted_label_lower"].dropna().unique())))
        print("\nMatrici di Confusione per ciascuna classe per ogni AJCC Stage:")
        print("==============================================================")
        for stage in stages:
            stage_df = df[df["AJCC_Stage"] == stage]
            print(f"\nAJCC Stage: {stage}")
            for cls in tutte_classi:
                TP_cls = stage_df[(stage_df["true_lower"] == cls) & (stage_df["predicted_label_lower"] == cls)].shape[0]
                FP_cls = stage_df[(stage_df["true_lower"] != cls) & (stage_df["predicted_label_lower"] == cls)].shape[0]
                FN_cls = stage_df[(stage_df["true_lower"] == cls) & (stage_df["predicted_label_lower"] != cls)].shape[0]
                TN_cls = stage_df[(stage_df["true_lower"] != cls) & (stage_df["predicted_label_lower"] != cls)].shape[0]
                confusion_cls = pd.DataFrame({
                    f"Predetto '{cls}'": [TP_cls, FP_cls],
                    f"Predetto 'Non {cls}'": [FN_cls, TN_cls]
                }, index=[f"True '{cls}'", f"True 'Non {cls}'"])
                print(f"\nClasse: {cls}")
                print(confusion_cls.to_string(justify="center"))
                print()  # Riga vuota per separazione

    # Calcolo delle medie per sensitivity e specificity
    avg_sensitivity = sum(m["Sensitivity"] for m in metrics_cls) / len(metrics_cls)
    avg_specificity = sum(m["Specificity"] for m in metrics_cls) / len(metrics_cls)

    # Calcolo della mediana per sensitivity e specificity
    med_sensitivity = np.median([m["Sensitivity"] for m in metrics_cls])
    med_specificity = np.median([m["Specificity"] for m in metrics_cls])

    # Salvataggio dei risultati in un file txt
    results_text = ""
    results_text += "Confusion Matrix Binaria (Normal vs Non Normal):\n"
    results_text += mat.to_string(justify="center") + "\n\n"
    results_text += f"Sensitivity (TP / (TP + FN)): {sensitivity_bin:>8.4f} (TP={TP_bin}, FN={FN_bin})\n"
    results_text += f"Specificity (TN / (TN + FP)): {specificity_bin:>8.4f} (TN={TN_bin}, FP={FP_bin})\n\n"

    results_text += "Confusion Matrix aggregata per AJCC Stage:\n"
    for agg in agg_list:
        results_text += (f"Stadio {str(agg['AJCC_Stage']):<3} -> "
                         f"Sensitivity: {agg['Sensitivity']:<8.4f}\n")
    results_text += "\n"

    results_text += "Metriche per classe:\n"
    for metric in metrics_cls:
        results_text += f"Classe {metric['Classe']:<12} -> Sensitivity: {metric['Sensitivity']:<8.4f}, Specificity: {metric['Specificity']:<8.4f}\n"
    results_text += "\n"
    results_text += (f"Media Sensitivity:  {avg_sensitivity:<8.4f}, Media Specificity:  {avg_specificity:<8.4f}\n")
    results_text += (f"Mediana Sensitivity: {med_sensitivity:<8.4f}, Mediana Specificity: {med_specificity:<8.4f}\n")

    output_file = "results/Acc_Sen_Spe_AUC.txt"
    with open(output_file, "w") as f:
        f.write(results_text)
    print(f"\nRisultati salvati in {output_file}")


def load_data():
    preds_df = pd.read_csv("results/preds_detail_tumor.csv")
    preds_df = preds_df.rename(columns={'predicted': 'predicted_label'})

    excel_path = "Data.xlsx"
    (
        _, _, x_test_df,
        y_test, y_train_enc, y_val_enc, y_test_enc,
        X_train, X_val, X_test,
        X_train_resh, X_val_resh, X_test_resh,
        y_train_ohe, y_val_ohe, y_test_ohe,
        le, n_classes, genes_len
    ) = prepare_data_for_models(excel_path)


    if "AJCC_Stage" not in x_test_df.columns:
        print("Attenzione: 'AJCC_Stage' non trovata in x_test_df.")

    x_test_df = x_test_df.copy()
    x_test_df.index.name = "index"

    merged_df = preds_df.merge(x_test_df[['AJCC_Stage']], left_on='index', right_index=True, how='left')

    predictions = merged_df.to_dict(orient="records")
    return predictions


if __name__ == "__main__":
    predictions = load_data()
    calcola_metriche_da_lista(predictions)
