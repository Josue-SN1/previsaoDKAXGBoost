# Análise Exploratória e XGBoost

Este projeto carrega arquivos CSV de sessões de pacientes, cria um proxy simples de risco de DKA por sessão e treina um modelo XGBoost.

## Como usar

1. Ative o ambiente virtual:

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Instale as dependências, se necessário:

```powershell
pip install -r requirements.txt
```

3. Abra `analise_exploratoria_xgboost_clean.ipynb` no VS Code ou Jupyter Notebook.

4. Execute todas as células.

## Dependências mínimas

- numpy
- pandas
- scikit-learn
- xgboost

## Arquivos principais

- `analise_exploratoria_xgboost_clean.ipynb`: notebook limpo e simplificado.
- `requirements.txt`: dependências mínimas.
- `run_clean_notebook.py`: script para executar o notebook programaticamente.

## Observações

- Não versionamos o `.venv/` nem os dados CSV.
- Se quiser adicionar um repositório remoto, use `git remote add origin <url>` e `git push -u origin master`.
