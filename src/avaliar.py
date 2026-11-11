import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (roc_auc_score, recall_score, precision_score, f1_score,
                              accuracy_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.preparo import carregar_dados, separar_treino_teste, FEATURES_NUMERICAS, FEATURES_CATEGORICAS, RAIZ, RANDOM_STATE

PASTA_REPORTS = RAIZ / "reports"
PASTA_FIG = PASTA_REPORTS / "figuras"
PASTA_FIG.mkdir(parents=True, exist_ok=True)

MLFLOW_URI = f"sqlite:///{RAIZ.as_posix()}/mlflow.db"
EXPERIMENTO = "doenca-cardiaca"

sns.set_theme(style="whitegrid")


def encontrar_melhor_run():
    mlflow.set_tracking_uri(MLFLOW_URI)
    # só runs de treino (têm a tag conjunto), a avaliação final não entra na disputa
    runs = mlflow.search_runs(experiment_names=[EXPERIMENTO], filter_string="tags.conjunto LIKE '%'",
                              order_by=["metrics.cv_roc_auc_mean DESC"], max_results=1)
    if runs.empty:
        raise ValueError("nenhum run de treino, rode src.treinar antes")
    return runs.iloc[0]


def escolher_limiar_recall(modelo, X_train, y_train, recall_minimo=0.90):
    # previsões out-of-fold no treino, o teste não entra nessa escolha
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    y_proba_oof = cross_val_predict(clone(modelo), X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    # o maior limiar que ainda segura o recall mínimo (quanto maior, menos falso alarme)
    limiar_escolhido = None
    for limiar in np.round(np.arange(0.05, 0.96, 0.01), 2):
        if recall_score(y_train, (y_proba_oof >= limiar).astype(int)) >= recall_minimo:
            limiar_escolhido = limiar

    if limiar_escolhido is None:
        raise ValueError(f"nenhum limiar chega a recall {recall_minimo}")
    return float(limiar_escolhido)


def intervalo_bootstrap(y_true, y_proba, limiar, n=2000):
    # teste tem só 184 pacientes, sem intervalo o número sozinho engana
    rng = np.random.default_rng(RANDOM_STATE)
    y_true = np.asarray(y_true)
    aucs, recalls = [], []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        if y_true[i].min() == y_true[i].max():
            continue
        aucs.append(roc_auc_score(y_true[i], y_proba[i]))
        recalls.append(recall_score(y_true[i], (y_proba[i] >= limiar).astype(int)))
    return {
        "roc_auc_ic95": [round(float(np.percentile(aucs, 2.5)), 3), round(float(np.percentile(aucs, 97.5)), 3)],
        "recall_ic95": [round(float(np.percentile(recalls, 2.5)), 3), round(float(np.percentile(recalls, 97.5)), 3)],
    }


def auc_por_centro_no_teste(y_test, y_proba, origem_teste):
    # o AUC geral mistura centros com prevalência de 36% a 93%, então olho cada um separado
    linhas = []
    for centro, idx in origem_teste.groupby(origem_teste).groups.items():
        y_c = y_test.loc[idx]
        p_c = pd.Series(y_proba, index=y_test.index).loc[idx]
        linhas.append({
            "centro": centro,
            "n": len(y_c),
            "doentes": int(y_c.sum()),
            "roc_auc": round(roc_auc_score(y_c, p_c), 3) if y_c.nunique() == 2 else np.nan,
        })
    return pd.DataFrame(linhas)


def avaliar_teste(modelo, X_test, y_test, limiar):
    y_proba = modelo.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= limiar).astype(int)

    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "y_proba": y_proba,
        "y_pred": y_pred
    }


def validacao_centro_fora(modelo, df, conjunto, limiar):
    # treina em 3 centros e testa no que ficou de fora. é diagnóstico, não validação limpa:
    # modelo, hiperparâmetros e limiar foram escolhidos com os 4 centros no treino
    df = df.drop_duplicates()
    features_num = FEATURES_NUMERICAS
    if conjunto == "completo":
        features_cat = FEATURES_CATEGORICAS
    else:
        features_cat = [c for c in FEATURES_CATEGORICAS if c not in ["vasos_coloridos", "talassemia"]]

    features = features_num + features_cat
    resultados = []

    # Suíça tem só 123 pacientes e 93% doentes, AUC pode oscilar muito
    for centro_teste in ["Cleveland", "Hungria", "VA Long Beach", "Suíça"]:
        df_treino = df[df["origem"] != centro_teste]
        df_teste = df[df["origem"] == centro_teste]

        if len(df_teste) < 20:
            continue

        X_tr = df_treino[features]
        y_tr = df_treino["doenca"]
        X_te = df_teste[features]
        y_te = df_teste["doenca"]

        m = clone(modelo).fit(X_tr, y_tr)
        y_proba = m.predict_proba(X_te)[:, 1]

        auc = roc_auc_score(y_te, y_proba)
        recall = recall_score(y_te, (y_proba >= limiar).astype(int))

        resultados.append({
            "centro_teste": centro_teste,
            "n_teste": len(df_teste),
            "roc_auc": round(auc, 3),
            "recall": round(recall, 3)
        })

    return pd.DataFrame(resultados)


def gerar_figuras(df_resultados, y_test, y_proba, y_pred, modelo, X_test, conjunto_vencedor):
    # 05: comparação de modelos (só conjunto vencedor)
    df_plot = df_resultados[df_resultados["conjunto"] == conjunto_vencedor].copy()
    df_plot = df_plot.sort_values("cv_roc_auc_mean")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = df_plot["cv_roc_auc_mean"]
    y = range(len(df_plot))
    xerr = df_plot["cv_roc_auc_std"]
    ax.barh(y, x, xerr=xerr, color="steelblue", capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(df_plot["modelo"])
    ax.set_xlabel("ROC AUC (média e desvio entre os 15 folds)")
    ax.set_title(f"Comparação de modelos (conjunto {conjunto_vencedor}, validação cruzada no treino)")
    ax.set_xlim(0.45, 1.0)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    plt.tight_layout()
    fig.savefig(PASTA_FIG / "05_comparacao_modelos.png", dpi=120)
    plt.close()

    # 06: curva ROC no teste
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Aleatório")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.set_title("Curva ROC no conjunto de teste")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PASTA_FIG / "06_roc_teste.png", dpi=120)
    plt.close()

    # 07: matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["sem doença", "com doença"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Matriz de confusão no teste")
    plt.tight_layout()
    fig.savefig(PASTA_FIG / "07_matriz_confusao_teste.png", dpi=120)
    plt.close()

    # 08: importância de permutação (colunas antes do one-hot)
    perm_imp = permutation_importance(modelo, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE,
                                      scoring="roc_auc")

    indices = perm_imp.importances_mean.argsort()[-10:]
    nomes_features = X_test.columns.tolist()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(indices)), perm_imp.importances_mean[indices], xerr=perm_imp.importances_std[indices],
            color="steelblue", capsize=3)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([nomes_features[i] for i in indices])
    ax.set_xlabel("Importância (queda no ROC AUC)")
    ax.set_title("Top 10 features por importância de permutação")
    plt.tight_layout()
    fig.savefig(PASTA_FIG / "08_importancia_permutacao.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENTO)

    melhor_run = encontrar_melhor_run()
    run_id = melhor_run["run_id"]
    conjunto_vencedor = melhor_run["tags.conjunto"]
    modelo_nome = melhor_run["tags.modelo"]

    print(f"Melhor run: {modelo_nome} | {conjunto_vencedor} (AUC={melhor_run['metrics.cv_roc_auc_mean']:.3f})")

    # no mlflow 3 o modelo tem id próprio, separado do run
    logado = mlflow.search_logged_models(experiment_ids=[melhor_run["experiment_id"]],
                                         filter_string=f"source_run_id = '{run_id}'", output_format="list")[0]
    modelo_uri = logado.model_uri
    modelo = mlflow.sklearn.load_model(modelo_uri)

    df = carregar_dados()
    X_train, X_test, y_train, y_test = separar_treino_teste(df)

    # em triagem, deixar um doente passar custa mais que um falso alarme
    limiar = escolher_limiar_recall(modelo, X_train, y_train, recall_minimo=0.90)
    print(f"Limiar: {limiar:.2f}")

    metricas = avaliar_teste(modelo, X_test, y_test, limiar)
    y_proba = metricas.pop("y_proba")
    y_pred = metricas.pop("y_pred")

    print("\nMétricas no teste:")
    for k, v in metricas.items():
        print(f"  {k}: {v:.3f}")

    ic = intervalo_bootstrap(y_test, y_proba, limiar)
    print(ic)

    # mesmo dedup do split, pra o índice bater com X_test
    origem_teste = df.drop_duplicates().reset_index(drop=True).loc[X_test.index, "origem"]
    df_centro_teste = auc_por_centro_no_teste(y_test, y_proba, origem_teste)
    df_centro_teste.to_csv(PASTA_REPORTS / "auc_por_centro_teste.csv", index=False)
    print(df_centro_teste.to_string(index=False))

    with mlflow.start_run(run_name="avaliacao_final"):
        mlflow.set_tags({"run_origem": run_id, "limiar": limiar})
        for k, v in metricas.items():
            mlflow.log_metric(k, v)

    client = MlflowClient(MLFLOW_URI)
    model_name = "doenca-cardiaca"

    mv = mlflow.register_model(modelo_uri, model_name)
    client.set_model_version_tag(model_name, mv.version, "limiar", str(limiar))
    client.set_registered_model_alias(model_name, "champion", mv.version)

    print(f"\nModelo registrado como '{model_name}' versão {mv.version} com alias 'champion'")

    df_centro = validacao_centro_fora(modelo, df, conjunto_vencedor, limiar)
    df_centro.to_csv(PASTA_REPORTS / "centro_fora.csv", index=False)
    print("\nCentro fora do treino:")
    print(df_centro.to_string(index=False))

    df_resultados = pd.read_csv(PASTA_REPORTS / "resultados_cv.csv")
    gerar_figuras(df_resultados, y_test, y_proba, y_pred, modelo, X_test, conjunto_vencedor)

    with open(PASTA_REPORTS / "metricas_teste.json", "w") as f:
        json.dump({"modelo": modelo_nome, "conjunto": conjunto_vencedor, "limiar": limiar, **metricas, **ic},
                  f, indent=2)
