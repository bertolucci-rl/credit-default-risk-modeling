# Case Técnico — Cientista de Dados Júnior (Datarisk)

Este projeto desenvolve um modelo preditivo para estimar a probabilidade de inadimplência, utilizando as bases de dados fornecidas pela Datarisk.

---

## 1. Estrutura do Projeto

- `notebook.ipynb` — Análise completa, preparação dos dados, modelagem e geração da submissão final;
- `requirements.txt` — Dependências necessárias;
- `submissao_case.csv` — Arquivo com as probabilidades previstas;
- `README.md` — Documentação do projeto.

A pasta `data/` **não está incluída**, conforme instruções do case.  
O avaliador deverá criá-la e inserir os arquivos .csv originais.

---

## 2. Como Executar o Projeto

### 2.1 Estrutura esperada

```
├── notebook.ipynb
├── requirements.txt
├── README.md
└── data/
```

### 2.2 Arquivos necessários dentro da pasta `data/`

- `base_cadastral.csv`;
- `base_info.csv`;  
- `base_pagamentos_desenvolvimento.csv`;  
- `base_pagamentos_teste.csv`.

### 2.3 Instalação das dependências

```
pip install -r requirements.txt
```

### 2.4 Execução do notebook

Execute o notebook de cima para baixo. Ele realizará automaticamente:

- leitura e consolidação das bases;
- tratamento de valores ausentes;
- criação de variáveis derivadas;
- pré-processamento com ColumnTransformer;
- modelagem com 3 algoritmos;
- comparação das métricas;
- validação cruzada;
- tuning leve de hiperparâmetros;
- geração da submissão `submissao_case.csv`.

---

## 3. Preparação dos Dados

Etapas principais:

- Consolidação das bases conforme o relacionamento fornecido;  
- Ajuste e padronização de tipos (datas, numéricas e categóricas);  
- Feature engineering:
  - SAFRA_ANO, SAFRA_MES;  
  - prazo entre emissão e vencimento;  
  - taxa relativa;  
- Tratamento de valores ausentes:
  - mediana (numéricas);  
  - `"NA"` (categóricas);  
- Codificação com OneHotEncoder;  
- Correção do desbalanceamento com `class_weight = 'balanced'`.  

---

## 4. Modelos Avaliados

Três algoritmos supervisionados foram testados:

- **Regressão Logística**;
- **Random Forest**;
- **XGBoost**.

Todos integrados no mesmo pipeline de pré-processamento.

---

## 5. Resultados

### 5.1 Desempenho na validação

| Modelo                | AUC-ROC | Log Loss |
|----------------------|---------|----------|
| Regressão Logística  | ~0.81   | ~0.53    |
| XGBoost              | ~0.94   | ~0.26    |
| **Random Forest**    | **~0.96** | **~0.11** |

### 5.2 Modelo final selecionado: **Random Forest Otimizado**

Motivos da escolha:

- Maior AUC-ROC entre todos os modelos testados;
- Menor Log Loss;
- Melhor desempenho na classe minoritária;
- Estabilidade comprovada via validação cruzada;
- Desempenho refinado via RandomizedSearchCV.

Resultados finais:

- **AUC-ROC ≈ 0.9624**;
- **Log Loss ≈ 0.1170**.

---

## 6. Submissão

O arquivo `submissao_case.csv` contém:

- `ID_CLIENTE`;
- `SAFRA_REF`;
- `PROBABILIDADE_INADIMPLENCIA`.

Formato conforme o solicitado no case.

---

## 7. Conclusões

- O desbalanceamento da variável-alvo exigiu métricas adequadas (AUC-ROC e Log Loss) e uso de `class_weight = 'balanced'`;
- A engenharia de atributos contribuiu para o aumento da performance; 
- O Random Forest se mostrou o modelo mais robusto, estável e eficaz;
- A validação cruzada confirmou que o desempenho não depende de um único split;  
- O ajuste leve via RandomizedSearchCV trouxe ganho adicional sem aumentar significativamente o custo computacional.

---

## 8. Reprodutibilidade

Para instalar as dependências:

```
pip install -r requirements.txt
```

Após isso, basta executar o notebook.

---

