import pandas as pd
import warnings

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from src.preparo import carregar_dados, separar_treino_teste, FEATURES_NUMERICAS, FEATURES_CATEGORICAS, RAIZ, RANDOM_STATE

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

PASTA_REPORTS = RAIZ / "reports"
PASTA_REPORTS.mkdir(exist_ok=True)

MLFLOW_URI = f"sqlite:///{RAIZ.as_posix()}/mlflow.db"
EXPERIMENTO = "doenca-cardiaca"

METRICAS = ["roc_auc", "recall", "precision", "f1", "accuracy"]

# o mlflow 3 salva com skops e recusa tipos fora da lista padrão, esses aqui são de sklearn/xgboost
TIPOS_CONFIAVEIS = ["numpy.dtype", "sklearn.tree._tree.Tree",
                    "sklearn.calibration._CalibratedClassifier", "sklearn.calibration._SigmoidCalibration",
                    "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]


def criar_preprocessador(conjunto):
    if conjunto == "completo":
        num = FEATURES_NUMERICAS
        cat = FEATURES_CATEGORICAS
    else:  # sem_vasos_talassemia
        num = FEATURES_NUMERICAS
        cat = [c for c in FEATURES_CATEGORICAS if c not in ["vasos_coloridos", "talassemia"]]

    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])


def definir_modelos():
    return {
        "dummy": (DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE), {}),
        "logistic": (LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
                     {"modelo__C": [0.1, 1, 10]}),
        "knn": (KNeighborsClassifier(),
                {"modelo__n_neighbors": [3, 5, 10], "modelo__weights": ["uniform", "distance"]}),
        # SVC(probability=True) foi depreciado no sklearn 1.9, a doc indica calibrar por fora
        "svc": (CalibratedClassifierCV(SVC(random_state=RANDOM_STATE), ensemble=False),
                {"modelo__estimator__C": [0.1, 1, 10], "modelo__estimator__kernel": ["rbf", "linear"]}),
        "tree": (DecisionTreeClassifier(random_state=RANDOM_STATE),
                 {"modelo__max_depth": [3, 5, 10, None], "modelo__min_samples_split": [2, 5]}),
        "random_forest": (RandomForestClassifier(random_state=RANDOM_STATE),
                          {"modelo__n_estimators": [50, 100], "modelo__max_depth": [5, 10, None],
                           "modelo__min_samples_split": [2, 5]}),
        "gradient_boosting": (GradientBoostingClassifier(random_state=RANDOM_STATE),
                              {"modelo__n_estimators": [50, 100], "modelo__learning_rate": [0.05, 0.1],
                               "modelo__max_depth": [3, 5]}),
        "xgboost": (XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
                    {"modelo__n_estimators": [50, 100], "modelo__learning_rate": [0.05, 0.1],
                     "modelo__max_depth": [3, 5]})
    }


def treinar_avaliar(X_train, y_train, conjunto):
    resultados = []
    modelos_dict = definir_modelos()

    for nome, (modelo, params) in modelos_dict.items():
        pipe = Pipeline([
            ("preprocessador", criar_preprocessador(conjunto)),
            ("modelo", modelo)
        ])

        if params:
            busca = GridSearchCV(pipe, params, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                                 scoring="roc_auc", n_jobs=-1)
            busca.fit(X_train, y_train)
            melhor_pipe = busca.best_estimator_
            melhor_params = busca.best_params_
        else:
            pipe.fit(X_train, y_train)
            melhor_pipe = pipe
            melhor_params = {}

        cv_results = cross_validate(
            melhor_pipe, X_train, y_train,
            cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE),
            scoring=METRICAS, n_jobs=-1, return_train_score=False
        )

        run_name = f"{nome} | {conjunto}"

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({"modelo": nome, "conjunto": conjunto})

            for k, v in melhor_params.items():
                mlflow.log_param(k.replace("modelo__", ""), v)

            metricas_log = {}
            for metrica in METRICAS:
                media = cv_results[f"test_{metrica}"].mean()
                desvio = cv_results[f"test_{metrica}"].std()
                mlflow.log_metric(f"cv_{metrica}_mean", media)
                mlflow.log_metric(f"cv_{metrica}_std", desvio)
                metricas_log[f"cv_{metrica}_mean"] = media
                metricas_log[f"cv_{metrica}_std"] = desvio

            exemplo = X_train.head(5)
            signature = infer_signature(exemplo, melhor_pipe.predict(exemplo))
            mlflow.sklearn.log_model(melhor_pipe, name="modelo", signature=signature, input_example=exemplo,
                                     skops_trusted_types=TIPOS_CONFIAVEIS)

            resultado = {"modelo": nome, "conjunto": conjunto, **metricas_log}
            resultados.append(resultado)

    return resultados


if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENTO)

    df = carregar_dados()
    X_train, X_test, y_train, y_test = separar_treino_teste(df)

    print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")

    todos_resultados = []

    for conjunto in ["completo", "sem_vasos_talassemia"]:
        print(conjunto)
        res = treinar_avaliar(X_train, y_train, conjunto)
        todos_resultados.extend(res)

    df_resultados = pd.DataFrame(todos_resultados).sort_values("cv_roc_auc_mean", ascending=False)
    df_resultados.to_csv(PASTA_REPORTS / "resultados_cv.csv", index=False)

    colunas = ["modelo", "conjunto", "cv_roc_auc_mean", "cv_roc_auc_std", "cv_recall_mean", "cv_f1_mean"]
    print(df_resultados[colunas].round(3).to_string(index=False))
