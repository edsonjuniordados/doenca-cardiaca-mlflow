import sys
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.preparo import RAIZ, FEATURES_NUMERICAS, FEATURES_CATEGORICAS

MLFLOW_URI = f"sqlite:///{RAIZ.as_posix()}/mlflow.db"
NOME_MODELO = "doenca-cardiaca"


def prever(caminho_csv):
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient(MLFLOW_URI)

    modelo = mlflow.sklearn.load_model(f"models:/{NOME_MODELO}@champion")
    versao = client.get_model_version_by_alias(NOME_MODELO, "champion")
    limiar = float(versao.tags["limiar"])

    df = pd.read_csv(caminho_csv)
    faltando = set(FEATURES_NUMERICAS + FEATURES_CATEGORICAS) - set(df.columns)
    if faltando:
        raise ValueError(f"colunas faltando no csv: {sorted(faltando)}")

    # o modelo espera as categorias já decodificadas ("masculino", "assintomático"...), como sai de src.preparo.limpar
    if pd.api.types.is_numeric_dtype(df["sexo"]):
        raise ValueError("sexo veio numérico, passe o csv por src.preparo.limpar antes")

    probs = modelo.predict_proba(df)[:, 1]

    resultado = df.copy()
    resultado["probabilidade_doenca"] = probs.round(3)
    resultado["predicao"] = (probs >= limiar).astype(int)
    return resultado


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("uso: python -m src.prever <caminho_csv>")
        sys.exit(1)

    print(prever(sys.argv[1]).to_string(index=False))
