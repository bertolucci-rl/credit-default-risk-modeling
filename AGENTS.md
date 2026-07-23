# Instruções do Projeto

Este repositório contém um case técnico para uma vaga de Cientista de
Dados Júnior.

## Arquivos protegidos

Nunca modificar:

- `case_datarisk_notebook_original.ipynb`
- arquivos dentro de `data/`
- arquivos dentro de `docs/`

## Arquivo principal

A versão de trabalho será:

- `case_datarisk.ipynb`

## Restrições

- Não usar a base de teste para treinamento ou seleção do modelo.
- Não usar `DATA_PAGAMENTO`, `DIAS_ATRASO` ou `TARGET` como features.
- Não criar features históricas com dados da própria safra ou do futuro.
- Preservar a ordem original da base de teste.
- Não adicionar complexidade incompatível com vaga júnior.
- Não modificar arquivos fora do escopo do prompt atual.
- Não instalar novas bibliotecas sem justificar.
- Não alterar ou remover os dados originais.
- Não incluir informações pessoais nos entregáveis.
- Não criar commits automaticamente.
- Não executar comandos destrutivos do Git.
- Antes de editar, apresentar um plano curto.
- Depois de editar, listar alterações e testes executados.

## Notebook

- O notebook deve executar do início ao fim.
- Cada seção deve conter Markdown coerente com o código e os resultados.
- Não manter métricas antigas que contradigam a execução atual.
- Não afirmar causalidade, prontidão para produção ou generalização garantida.