import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_RAW = RAIZ / "data" / "raw" / "heart_disease_uci.csv"
RANDOM_STATE = 42

# origem e num não são features: origem correlaciona com alvo pela seleção de pacientes
# por centro, num é o alvo original (0-4, apenas usamos pra criar doenca binário)
FEATURES_NUMERICAS = ["idade", "pressao_repouso", "colesterol", "freq_max", "depressao_st"]
FEATURES_CATEGORICAS = ["sexo", "tipo_dor_peito", "glicemia_alta", "ecg_repouso",
                        "angina_exercicio", "inclinacao_st", "vasos_coloridos", "talassemia"]


def carregar_bruto():
    return pd.read_csv(CAMINHO_RAW)


def limpar(df):
    df = df.copy()

    df["sexo"] = df["sexo"].map({1: "masculino", 0: "feminino"})

    df["tipo_dor_peito"] = df["tipo_dor_peito"].map({
        1: "angina típica",
        2: "angina atípica",
        3: "dor não anginosa",
        4: "assintomático"
    })

    df["ecg_repouso"] = df["ecg_repouso"].map({
        0: "normal",
        1: "anormalidade ST-T",
        2: "hipertrofia VE"
    })

    df["inclinacao_st"] = df["inclinacao_st"].map({
        1: "ascendente",
        2: "plana",
        3: "descendente"
    })

    df["talassemia"] = df["talassemia"].map({
        3: "normal",
        6: "defeito fixo",
        7: "defeito reversível"
    })

    # colesterol e pressao 0 são impossíveis biologicamente, tratados como ausente
    # (todos 123 registros da Suíça têm colesterol 0, registraram ausente assim)
    df.loc[df["colesterol"] == 0, "colesterol"] = np.nan
    df.loc[df["pressao_repouso"] == 0, "pressao_repouso"] = np.nan

    # num vai de 0 a 4 (grau de estreitamento), qualquer grau conta como doença
    df["doenca"] = (df["num"] > 0).astype(int)

    return df


def carregar_dados():
    df = carregar_bruto()
    df = limpar(df)
    return df


def separar_treino_teste(df):
    # 2 linhas duplicadas exatas, ver notebook de exploração
    df = df.drop_duplicates().reset_index(drop=True)

    features = FEATURES_NUMERICAS + FEATURES_CATEGORICAS
    X = df[features]
    y = df["doenca"]

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
