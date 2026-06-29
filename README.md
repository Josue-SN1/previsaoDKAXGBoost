# Análise Exploratória e XGBoost

Este diretório contém a análise exploratória para predição de risco de DKA usando janelas temporais e XGBoost.

Importante: considere apenas esta pasta (`AnaliseExploratória`) como projeto ativo. A subpasta `TCC/` aqui dentro é uma cópia antiga e não deve ser usada.

## Estrutura principal

- `analise_temporal_risco_dka.ipynb`: notebook principal da análise.
- `requirements.txt`: dependências Python.

## Pré-requisitos

- Python 3.10+ (recomendado)
- PowerShell (Windows)

## Configuração do ambiente

1. Entre na pasta do projeto:

```powershell
cd "c:\Users\josuc\OneDrive - PUCRS - BR\PUCRS\2026_01\TCC\AnaliseExploratória"
```

2. Crie o ambiente virtual (caso necessário):

```powershell
python -m venv .venv
```

3. Ative o ambiente:

```powershell
.\.venv\Scripts\Activate.ps1
```

4. Instale as dependências:

```powershell
pip install -r requirements.txt
```

## Como rodar o projeto

### Opção A: Notebook (recomendado)

1. Abra `analise_temporal_risco_dka.ipynb` no VS Code.
2. Selecione o kernel da `.venv`.
3. Execute as células na ordem.

Saídas esperadas:
- Tabelas de distribuição de `risk_score`.
- Métricas (MAE, RMSE, etc.).
- Gráficos comparativos 7 dias vs 14 dias.
- Pré-visualização opcional de um CSV processado.
- Geração opcional de `session_labels.csv` dentro do próprio notebook.

## Como rodar os testes

No estado atual do projeto, não existe suíte de testes automatizados (`pytest`) versionada.

A validação recomendada é:

1. Executar todas as células do `analise_temporal_risco_dka.ipynb` sem erro no kernel.
2. Verificar se o notebook gera as tabelas, métricas e gráficos esperados.
3. Verificar se `session_labels.csv` foi gerado ou atualizado quando a célula opcional de resumo por sessão for executada.

## Possíveis problemas

- Erro de pacote ausente: execute novamente `pip install -r requirements.txt`.
- Kernel errado no notebook: troque para o Python da `.venv`.
- Caminho de dados não encontrado: confirme se as pastas `Ohio2018_processed/` e `Ohio2020_processed/` existem dentro de `AnaliseExploratória`.
