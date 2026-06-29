# Status do Projeto

## Escopo ativo

Considerar apenas esta pasta `AnaliseExploratória` como projeto ativo.
A subpasta `TCC/` dentro dela é uma cópia antiga e deve ser ignorada.

## Estado atual

- Notebook principal: `analise_temporal_risco_dka.ipynb`.
- O notebook foi ajustado para usar a divisão original de treino e teste do OhioDataset.
- O fluxo principal usa `risk_score` como alvo, sem depender do antigo proxy label.
- A lógica temporal usa atributos da janela `t` para prever o `risk_score` da janela `t + 1`.
- As funções auxiliares de inspeção de CSV e resumo por sessão foram incorporadas ao notebook.
- Os scripts redundantes `generate_analysis.py`, `create_session_labels.py` e `inspect_data.py` foram removidos.
- O `README.md` foi atualizado para um fluxo centrado apenas no notebook.

## Estrutura do notebook

1. Configuração do ambiente e definição dos diretórios de treino e teste.
2. Funções auxiliares de carregamento, preprocessamento, engenharia de atributos e criação de janelas temporais.
3. Etapas opcionais de inspeção de arquivo processado e resumo por sessão.
4. Modelagem temporal com janelas de 7 dias.
5. Comparação entre janelas de 7 e 14 dias.
6. Visualizações.
7. Resumo executivo e conclusões.

## Decisões importantes

- Evitar data leakage preservando a separação original entre `train/` e `test/` do OhioDataset.
- Tratar `missing_cbg` como indicador binário e preencher `NaN` com `0` antes das agregações.
- Manter a execução do notebook coerente de cima para baixo, com as funções definidas antes do primeiro uso.
- Posicionar as etapas opcionais antes da modelagem principal para manter a narrativa adequada ao TCC.

## Observações de execução

- A análise com janela de 7 dias foi executada com sucesso.
- A análise com janela de 14 dias pode não gerar janelas suficientes no conjunto de teste; o notebook já trata esse caso sem falhar.
- O projeto ainda contém artefatos gerados, como `session_labels.csv` e `temporal_analysis.png`.

## Próximos passos possíveis

1. Remover artefatos gerados caso a pasta deva conter apenas código-fonte e documentação.
2. Revisar o texto do notebook para deixá-lo mais acadêmico e alinhado ao TCC.
3. Refinar títulos, explicações e conclusões para a versão final do trabalho.

## Como retomar em outra sessão

Informar algo como:

> Continue a partir do arquivo `STATUS.md` e do notebook `analise_temporal_risco_dka.ipynb`.

