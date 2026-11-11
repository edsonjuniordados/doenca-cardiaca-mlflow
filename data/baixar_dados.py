import urllib.request
import zipfile
from pathlib import Path
from io import BytesIO
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
URL_UCI = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
CAMINHO_RAW = RAIZ / "data" / "raw" / "heart_disease_uci.csv"

ARQUIVOS_PROCESSADOS = [
    ("processed.cleveland.data", "Cleveland"),
    ("processed.hungarian.data", "Hungria"),
    ("processed.switzerland.data", "Suíça"),
    ("processed.va.data", "VA Long Beach")
]

COLUNAS = ["idade", "sexo", "tipo_dor_peito", "pressao_repouso", "colesterol",
           "glicemia_alta", "ecg_repouso", "freq_max", "angina_exercicio",
           "depressao_st", "inclinacao_st", "vasos_coloridos", "talassemia", "num"]


def baixar_e_processar():
    if CAMINHO_RAW.exists():
        print(f"Arquivo já existe: {CAMINHO_RAW}")
        return

    print(f"Baixando dados de {URL_UCI}")
    with urllib.request.urlopen(URL_UCI) as resp:
        zip_bytes = BytesIO(resp.read())

    dfs = []
    with zipfile.ZipFile(zip_bytes) as z:
        for arquivo, origem in ARQUIVOS_PROCESSADOS:
            conteudo = z.read(arquivo).decode("utf-8")
            linhas = [linha.strip() for linha in conteudo.split("\n") if linha.strip()]

            dados = []
            for linha in linhas:
                valores = [v.strip() if v.strip() != "?" else None for v in linha.split(",")]
                dados.append(valores)

            df = pd.DataFrame(dados, columns=COLUNAS)
            df["origem"] = origem
            dfs.append(df)

    df_completo = pd.concat(dfs, ignore_index=True)

    for col in COLUNAS:
        df_completo[col] = pd.to_numeric(df_completo[col], errors="coerce")

    CAMINHO_RAW.parent.mkdir(parents=True, exist_ok=True)
    df_completo.to_csv(CAMINHO_RAW, index=False)
    print(f"Salvo {len(df_completo)} registros em {CAMINHO_RAW}")


if __name__ == "__main__":
    baixar_e_processar()
