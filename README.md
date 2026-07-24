# Case Técnico — Cientista de Dados Júnior (Datarisk)

Este projeto estima a probabilidade de uma cobrança ser paga com atraso de cinco dias ou mais. A unidade de previsão é uma cobrança, e a saída é um valor contínuo entre 0 e 1.

## Estrutura dos arquivos

```text
case_datarisk.ipynb   # análise, preparação, modelagem e submissão
README.md             # documentação da solução
requirements.txt      # dependências com versões
submissao_case.csv    # probabilidades para a base de teste
data/                 # bases originais fornecidas, não incluídas na entrega
```

O notebook original, o PDF do case e as bases não fazem parte da pasta de entrega.

## Ambiente

- Python 3.12.5

Instalação:

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Estrutura esperada da pasta `data`

Antes da execução, crie a pasta `data/` ao lado do notebook e inclua:

```text
data/
├── base_cadastral.csv
├── base_info.csv
├── base_pagamentos_desenvolvimento.csv
└── base_pagamentos_teste.csv
```

Os arquivos são lidos com separador `;`.

## Execução

Abra `case_datarisk.ipynb` e execute todas as células em ordem, a partir de um kernel reiniciado. O notebook:

1. lê e valida as quatro bases;
2. converte datas e consolida as informações;
3. constrói o target;
4. realiza a análise exploratória;
5. cria features atuais e históricas;
6. compara modelos em um holdout temporal;
7. refina de forma contida o XGBoost selecionado;
8. treina o pipeline final com todo o desenvolvimento;
9. recria `submissao_case.csv`.

## Definição do target

```text
DIAS_ATRASO = DATA_PAGAMENTO - DATA_VENCIMENTO
TARGET = 1 quando DIAS_ATRASO >= 5
TARGET = 0 caso contrário
```

O desenvolvimento não possui pagamento ou vencimento ausente. Datas cronologicamente inconsistentes são reportadas e preservadas, pois o dicionário não fornece uma regra de correção.

## Resumo da EDA

- 77.414 cobranças e 1.248 clientes no desenvolvimento;
- target positivo em 7,02% das cobranças;
- variação mensal da inadimplência entre as safras observadas;
- múltiplas cobranças por cliente e forte sobreposição de clientes entre desenvolvimento e teste;
- assimetria em valor a pagar e renda;
- valores ausentes em renda e número de funcionários;
- diferenças descritivas entre desenvolvimento e teste, sem teste formal de drift;
- `FLAG_PF` interpretada conforme o dicionário: `X` representa pessoa física e ausência representa pessoa jurídica.

## Features

As features incluem:

- ano e mês da safra;
- prazo entre emissão e vencimento;
- tempo desde o cadastro;
- valor sobre renda;
- valor por funcionário;
- tipo de pessoa;
- variáveis originais da cobrança, cadastro e informação mensal;
- quantidade histórica de cobranças;
- quantidade histórica de inadimplências;
- taxa histórica de inadimplência;
- flag de cliente sem histórico.

O histórico é agregado por cliente e safra e deslocado antes de voltar às cobranças. A safra atual não participa das próprias features. Para o teste, o histórico usa somente pagamentos observados no desenvolvimento.

`ID_CLIENTE`, `_ROW_ID`, datas brutas, `DATA_PAGAMENTO`, `DIAS_ATRASO` e `TARGET` não são usados como features diretas.

Valores ausentes são tratados dentro dos pipelines. A Regressão Logística utiliza padronização; modelos de árvore não utilizam escala.

## Validação temporal

- treino: agosto de 2018 a março de 2021;
- validação: abril a junho de 2021;
- 70.012 linhas no treino;
- 7.402 linhas na validação.

O histórico da validação é congelado no final do treino. A base de teste não é utilizada para treinamento, seleção ou avaliação.

## Modelos avaliados

- Regressão Logística;
- Random Forest sem pesos;
- Random Forest com `class_weight="balanced"`, apenas como comparação;
- XGBoost sem pesos.

As métricas principais são Log Loss e ROC-AUC. Average Precision é usada como métrica complementar.

| Modelo | Configuração | ROC-AUC | Log Loss | Average Precision |
|---|---|---:|---:|---:|
| XGBoost | sem pesos | 0,929133 | 0,133252 | 0,580801 |
| Random Forest | sem pesos | 0,920540 | 0,144153 | 0,571888 |
| Regressão Logística | sem pesos | 0,856369 | 0,170143 | 0,432904 |
| Random Forest | balanceado | 0,928309 | 0,339570 | 0,574022 |

## Modelo escolhido

O candidato final é um XGBoost com:

```text
n_estimators=250
learning_rate=0.05
max_depth=3
subsample=0.8
colsample_bytree=0.8
objective="binary:logistic"
eval_metric="logloss"
random_state=0
```

Foram comparadas quatro configurações. A versão com 400 árvores e `learning_rate=0.03` reduziu o Log Loss em apenas 0,000136, abaixo do ganho mínimo de 0,002 definido antes do refinamento. Por isso, a baseline mais simples foi mantida.

Na validação, a probabilidade média prevista foi 5,22%, frente a uma taxa observada de 6,24%.

## Submissão

O notebook gera `submissao_case.csv` sem índice e com exatamente:

```text
ID_CLIENTE
SAFRA_REF
PROBABILIDADE_INADIMPLENCIA
```

`_ROW_ID` preserva a ordem original da base de teste. O notebook verifica quantidade de linhas, nomes das colunas, ordem, valores ausentes e intervalo das probabilidades.

## Limitações

- as métricas vêm de um único holdout temporal;
- não há garantia de desempenho em períodos posteriores;
- a probabilidade média subestima a taxa observada no holdout em aproximadamente 1,02 ponto percentual;
- algumas datas inconsistentes foram preservadas;
- uma categoria de DDD aparece apenas no teste;
- importâncias do XGBoost são preditivas e não representam causalidade;
- a solução não foi validada para uso em produção.

## Reprodutibilidade

- todas as sementes aleatórias utilizam `random_state=0`;
- imputação e codificação são ajustadas dentro dos pipelines;
- o notebook deve ser executado do início ao fim;
- a submissão é recriada automaticamente a partir das quatro bases originais.
