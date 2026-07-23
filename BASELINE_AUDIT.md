# BASELINE_AUDIT — Auditoria da Solução Anterior

**Escopo:** auditoria somente-leitura da *solução anterior* (`case_datarisk_notebook_original.ipynb`), confrontada com o enunciado (`docs/Case DS Júnior 2025.pdf`), o `README.md`, os quatro CSVs em `data/` e os artefatos de entrega. Nenhum arquivo existente foi modificado.

**Ambiente da auditoria:** Python 3.12.5; venv com `pandas 3.0.5`, `numpy 2.5.1`, `scikit-learn 1.9.0`, `xgboost 3.3.0`. (Para ler o PDF foi instalado `pypdf` apenas no venv de auditoria; não altera entregáveis.)

**Método:** leitura integral do PDF e do notebook original; releitura dos 4 CSVs; reexecução do notebook original em kernel limpo (diretório isolado, sem sobrescrever entregáveis); reprodução independente do pipeline e testes controlados de vazamento (split aleatório × por cliente × temporal).

---

## Resumo Executivo — achados que definem a nota

| # | Achado | Classe |
|---|--------|--------|
| A | **AUC 0,96 é inflado por vazamento de cliente** (split aleatório coloca o mesmo cliente em treino e validação). Sob split honesto: **0,78 (por cliente)** / **0,91 (temporal)**. | 🔴 Bloqueador |
| B | **Validação não-temporal** num problema temporal (teste = 2021‑07..11, estritamente após o desenvolvimento 2018‑08..2021‑06). Sem holdout/`GroupKFold` por cliente. | 🔴 Bloqueador |
| C | **Notebook original não executa de ponta a ponta** no ambiente instalado (erro em `RocCurveDisplay(..., color=...)`, sklearn ≥1.9), e `requirements.txt` (sklearn 1.6.1) **diverge do ambiente**. | 🔴 Bloqueador |
| D | **Afirmações incompatíveis com as saídas**: RF alegado como "melhor na classe minoritária" (recall 0,65) quando o XGBoost tem recall 0,86; "pronto para produção" e "excelente calibração" sem sustentação. | 🔴 Bloqueador |
| E | `TAXA_RELATIVA` é **redundante** (corr. 0,94 com `VALOR_A_PAGAR`) e mal nomeada (é custo absoluto, não relativo). | 🟠 Importante |
| F | **README aponta `notebook.ipynb`**, arquivo inexistente; `submissao_case.csv` atual **não** foi gerado pelo notebook original. | 🟠 Importante |

Legenda: 🔴 Bloqueador · 🟠 Importante · 🟢 Opcional.

---

## 1. Estrutura Atual do Projeto

**Arquivos existentes (versionados / relevantes):**
- `case_datarisk_notebook_original.ipynb` — solução anterior (177 células; 115 md / 62 código; execução fora de ordem: `exec_count` 37→174).
- `case_datarisk_notebook.ipynb` — **cópia reexecutada** do original (mesmas 177 células; `exec_count` limpo 1‑62).
- `case_datarisk.ipynb` — versão de trabalho nova, em progresso (38 células). *(fora do escopo desta auditoria; ver §8)*
- `README.md`, `requirements.txt`, `requirements-original.txt`, `AGENTS.md`, `CHECKLIST.md`
- `docs/Case DS Júnior 2025.pdf`, `data/*.csv` (4), `submissao_case.csv`, `.gitignore`, `.venv/`, `.agents/` (vazia)

**Arquivos realmente necessários para a entrega (conforme PDF §Entregáveis):** um notebook reprodutível, `requirements.txt` com versões, `submissao_case.csv`, e um documento de decisões (o README serve). Ou seja: **1 notebook + requirements + submissão + README**.

**Arquivos incompletos ou desnecessários:**
- `requirements-original.txt` — dump completo de `pip freeze` em **UTF‑16** (ilegível como texto simples), com ~centenas de pacotes (jupyter, anyio, etc.). Desnecessário na entrega e conflita com `requirements.txt`. 🟠
- Três notebooks coexistindo (`_original`, `_notebook`, o novo) — ambiguidade sobre qual é a entrega. 🟠
- `.agents/` vazia — inócua. 🟢
- `AGENTS.md`/`CHECKLIST.md` — artefatos internos de processo; **não devem ir no .zip** (anonimato/limpeza). 🟢

**Divergências entre nomes do README e nomes reais:** 🟠 (Importante)
- README §1 e §2.1 citam **`notebook.ipynb`** — **não existe** arquivo com esse nome. Os reais são `case_datarisk_notebook_original.ipynb` / `case_datarisk.ipynb`.
- README diz que `data/` "não está incluída"; de fato os CSVs existem localmente e estão em `.gitignore` (`data/`, `*.csv`) — coerente com a entrega, mas o texto e a árvore local divergem.

**Dependências ausentes ou desnecessárias:**
- `requirements.txt` lista 6 pacotes (pandas 2.2.2, numpy 2.2.4, scikit‑learn 1.6.1, xgboost 3.0.0, matplotlib 3.9.2, seaborn 0.13.2). **Ausente:** nada crítico; **divergente:** o venv tem versões bem mais novas (sklearn 1.9.0, pandas 3.0.5) — ver §8/Reprodutibilidade. `jupyter`/`ipykernel` não constam (necessários para rodar o `.ipynb`, embora implícitos).

---

## 2. Dados

| Base | Shape | Clientes únicos | Chave esperada | Duplicações na chave |
|------|-------|-----------------|----------------|----------------------|
| `base_cadastral` | (1315, 8) | 1315 | `ID_CLIENTE` | 0 |
| `base_info` | (24401, 4) | 1336 | (`ID_CLIENTE`,`SAFRA_REF`) | 0 |
| `base_pagamentos_desenvolvimento` | (77414, 7) | 1248 | linha = cobrança (N por ID/SAFRA) | 1 linha idêntica |
| `base_pagamentos_teste` | (12275, 6) | 976 | linha = cobrança | 11 linhas idênticas |

- **Período de desenvolvimento:** `SAFRA_REF` de **2018‑08 a 2021‑06** (35 meses).
- **Período de teste:** `SAFRA_REF` de **2021‑07 a 2021‑11** (5 meses) — **estritamente posterior** ao desenvolvimento. (`base_info` cobre 2018‑09..2021‑12.) → o problema é **temporal**; implica validação temporal (ver §7).
- **Chaves:** cadastral única por `ID_CLIENTE` (0 dup); info única por (`ID_CLIENTE`,`SAFRA_REF`) (0 dup). Confirmado no notebook (células [42]/[43]) e reproduzido.
- **Duplicações relevantes:** há **11 linhas 100% idênticas na base de teste** (e 1 no dev). Não são erradas por si (cobranças repetidas), mas geram previsões idênticas repetidas — vale registrar.
- **Valores ausentes relevantes:**
  - `FLAG_PF`: **94,98%** ausente em cadastral (≈99,7% após merge). *(Pelo dicionário, ausência = pessoa jurídica; portanto o "missing" carrega sinal — ver §6.)*
  - `NO_FUNCIONARIOS` 5,13% (info) → ~9,8% no dev consolidado; `RENDA_MES_ANTERIOR` ~7,9%; `DDD` ~9,6%; `PORTE` ~3,2%; `SEGMENTO_INDUSTRIAL` ~1,8%; `DOMINIO_EMAIL` ~1,2%.
  - **`VALOR_A_PAGAR` ausente**: 1,5% no dev e 1,07% no teste — é a variável financeira central e mesmo assim tem faltantes (imputados pela mediana; ver §6/§7).
- **Datas inválidas ou ausentes:**
  - Após `to_datetime(errors='coerce')`, **nenhuma data vira NaT** (0 ausências) nas três colunas do dev.
  - **`DATA_VENCIMENTO` chega a 2027‑03‑31** (18 registros com vencimento > 2022; mínimo 2017‑11‑27, anterior à 1ª safra). São datas suspeitas/outliers **não tratadas** — distorcem `PRAZO_EMISSAO_VENC`. 🟠
  - `DATA_PAGAMENTO`: **0 ausentes** no dev (relevante para §3).

---

## 3. Target

- **Fórmula (célula [29]):** `DIAS_ATRASO = (DATA_PAGAMENTO − DATA_VENCIMENTO).dt.days`; `TARGET = (DIAS_ATRASO >= 5).astype(int)`.
- **Regra de 5 dias:** ✅ **correta** — o PDF define inadimplência como "5 dias **ou mais** de atraso"; `>= 5` está certo. Caso‑limite atraso exatamente = 5 → TARGET 1 (correto; 955 registros com atraso=5).
- **Proporção do TARGET:** **7,02%** inadimplentes (5 436 de 77 414); 92,98% adimplentes. Reproduzido.
- **Tratamento de pagamentos ausentes:** **não há** `DATA_PAGAMENTO` ausente no dev (0), então na prática nada é perdido. **Porém** o código é frágil: se houvesse NaT, `DIAS_ATRASO` seria NaN e `(NaN >= 5)` → `False` → **TARGET 0 silencioso** (rotularia um não‑pagamento como adimplente). Não afeta este dataset, mas é uma armadilha latente. 🟢
- **Casos limítrofes:** 7 906 registros com atraso negativo (pago antes do vencimento) → TARGET 0 (correto); 60 742 com atraso=0. Distribuição perto do corte é contínua (…3:525, 4:305, 5:955, 6:700…), sem descontinuidade que sugira erro de corte.
- **Erros/ambiguidades:** o comentário do código diz `# 1 se houve atraso, 0 caso contrário` — impreciso; deveria dizer "atraso ≥ 5 dias". `DIAS_ATRASO` é criado em `base_dev` e permanece em `dev_full`/`X` (embora **não** usado como feature — ver §6). 🟢

---

## 4. Consolidação

- **Relacionamentos (célula [35]/[36]):** `base_dev.merge(base_cadastral, on='ID_CLIENTE')` depois `.merge(base_info, on=['ID_CLIENTE','SAFRA_REF'])`; idem para teste.
- **Tipos de join:** ambos **`how='left'`** com a base de pagamentos como tabela‑fato. Adequado.
- **Preservação de linhas:** ✅ 77 414 → 77 414 (dev) e 12 275 → 12 275 (teste). Verificado no notebook (célula [41]) e reproduzido.
- **Risco de multiplicação:** **nulo** — cadastral é 1 linha/`ID` (0 dup) e info é 1 linha/(`ID`,`SAFRA`) (`groupby().size().max() == 1`). Corretamente checado nas células [42]/[43].
- **Cobertura cadastral/mensal:** **todos os 1 248 clientes do dev estão na cadastral** (0 falhas de merge). Os faltantes pós‑merge vêm de **NaNs próprios da cadastral/info** (ex.: `DDD`, `RENDA`), não de chaves órfãs. (21 clientes de `base_info` não estão na cadastral, mas nenhum deles aparece no dev — irrelevante.)

*Observação:* a consolidação é sólida. O problema **não** está nos merges e sim no que vem depois (validação — §7).

---

## 5. EDA

**Análises existentes:** distribuição do TARGET (contagem + barra); ranking de ausentes; `describe()` das numéricas; `value_counts` das categóricas; boxplots de 4 numéricas por TARGET; heatmap de correlação; pairplot.

- **Gráficos úteis:** boxplots por TARGET e a barra do TARGET (mostram o desbalanceamento e a baixa separação).
- **Gráficos redundantes:** 🟠 heatmap de correlação **+** `pairplot` das **mesmas 4** numéricas dizem a mesma coisa (independência) — o `pairplot` (20 subplots, célula [66]) é pesado e redundante.
- **Interpretações incorretas / não sustentadas:**
  - [69]/[174] afirmam "padrões consistentes entre características cadastrais e comportamento" — **a própria EDA não cruza categóricas × TARGET**, então essa conclusão não é sustentada pelos gráficos apresentados. 🟠
  - "Não há separação evidente pelas variáveis individuais" é honesto e **contradiz** o posterior AUC 0,96 sem que ninguém questione a discrepância.
- **Ausência de análises importantes:** 🔴/🟠
  - **Nenhuma análise temporal** da taxa de inadimplência (por safra/ano) — essencial dado que o teste é futuro.
  - **Nenhuma análise da estrutura por cliente** (repetição de `ID_CLIENTE`, taxa de default por cliente) — é exatamente o que revelaria o vazamento do §7. (Medido nesta auditoria: desvio‑padrão da taxa de default por cliente = 0,232; 52% dos clientes têm todos os pagamentos adimplentes.)
  - Sem relação categóricas × TARGET, sem tratamento de outliers/datas futuras, sem discussão do sinal de `FLAG_PF`.

---

## 6. Features

**Features criadas (célula [78]):** `PRAZO_EMISSAO_VENC` (dias emissão→vencimento), `SAFRA_ANO`, `SAFRA_MES`, `TAXA_RELATIVA`. Preditores no modelo: `VALOR_A_PAGAR, TAXA, RENDA_MES_ANTERIOR, NO_FUNCIONARIOS, PRAZO_EMISSAO_VENC, SAFRA_ANO, SAFRA_MES, TAXA_RELATIVA` + categóricas `SEGMENTO_INDUSTRIAL, DOMINIO_EMAIL, PORTE, CEP_2_DIG`.

- **Redundantes:** 🟠 `TAXA_RELATIVA = TAXA × VALOR_A_PAGAR` tem **corr. 0,94 com `VALOR_A_PAGAR`** (TAXA tem variância baixa, ~5–7). É quase uma cópia reescalada — pouca informação nova.
- **Possíveis vazamentos:** `DIAS_ATRASO` e `DATA_PAGAMENTO` **permanecem em `X`** (só `TARGET` é removido em [88]). **Não vazam para o modelo** porque o `ColumnTransformer` lista as colunas explicitamente e usa `remainder='drop'` (verificado). Ainda assim é **frágil** — qualquer mudança para `remainder='passthrough'` ou uso de `X` cru vazaria o alvo. Conforme `AGENTS.md`, essas colunas nem deveriam estar em `X`. 🟠
- **Avaliação de `TAXA_RELATIVA`:** 🟠 (a) **mal nomeada** — o markdown [77] diz "custo *relativo* da taxa", mas o cálculo é **custo absoluto** dos juros (`taxa%×valor`); um custo *relativo* não dependeria do valor. (b) **redundante** (item acima). (c) não há evidência de ganho preditivo. Recomenda‑se remover ou substituir por algo genuinamente novo (ex.: juros normalizado pela renda).
- **Informações relevantes não utilizadas:** 🟠
  - **Histórico de comportamento do cliente** com defasagem temporal correta (taxa de atraso passada, nº de cobranças anteriores) — o PDF pede explicitamente explorar o "histórico de comportamento". Ausente.
  - **Idade do cadastro** (`hoje/emissão − DATA_CADASTRO`), **região via `DDD`/`CEP_2_DIG`** como sinal, e **`FLAG_PF`** transformado em binário PF/PJ (ausência = PJ) em vez de descartado.
  - `PRAZO_EMISSAO_VENC` é contaminado pelos vencimentos de 2027 (outliers não tratados).

---

## 7. Modelagem

- **Estratégia de split (célula [98]):** `train_test_split(test_size=0.2, stratify=y, random_state=0)` — **split aleatório por linha**. 🔴 **Bloqueador.**
- **Modelos:** Regressão Logística (baseline), Random Forest (300 árvores, `min_samples_split=5`), XGBoost (400 árv., `scale_pos_weight`). Todos em `Pipeline` com o pré‑processamento.
- **Preprocessing:** `StandardScaler` (num) + `OneHotEncoder(handle_unknown='ignore')` (cat) via `ColumnTransformer`. Adequado.
- **Imputação:** 🟠 feita **fora do pipeline e antes do split** (célula [75]), com a **mediana calculada sobre todo o dev** (inclui a futura validação) → leve vazamento de estatística treino↔validação; e a mesma mediana é reaplicada no teste (direção correta).
- **Desbalanceamento:** `class_weight='balanced'` (LR/RF) e `scale_pos_weight` (XGB). Correto para o problema.
- **Métricas:** AUC‑ROC, Log Loss, `classification_report`, matriz de confusão, curvas ROC. Apropriadas para probabilidade + desbalanceamento.
- **Validação cruzada (célula [148]):** `cross_val_score(cv=5, scoring='roc_auc')` — **KFold aleatório**, sobre os mesmos dados com clientes repetidos → herda o mesmo vazamento; o "0,96 estável" é **falsa estabilidade**.
- **Tuning:** `RandomizedSearchCV` (12 iter, cv=5) — ganho marginal (0,9618→0,9624), otimizando uma **métrica já contaminada**.

### Vazamento — evidência quantitativa (reproduzida nesta auditoria)
Mesmo pipeline (RF), variando **apenas** a estratégia de split:

| Estratégia de validação | AUC‑ROC |
|---|---|
| Split aleatório por linha (**como no notebook**) | **0,9624** ✅ reproduz o notebook |
| Split **por cliente** (`GroupShuffleSplit` no `ID_CLIENTE`) | **0,7828** |
| Split **temporal** (treino < 2021, validação 2021 — espelha a tarefa real) | **0,9063** |

**Conclusão:** o AUC 0,96 mede em grande parte a **memorização da propensão de cada cliente** (o mesmo `ID_CLIENTE` aparece em treino e validação com features quase idênticas e alvos correlacionados). O desempenho **honesto** de generalização é **~0,78–0,91**. Como a *base de teste é de clientes/meses futuros*, a estimativa relevante é a temporal (~0,91), não 0,96.

- **Coerência problema probabilístico × avaliação:** 🟠 o case pede **apenas probabilidade** (0–1), mas o notebook gera `classification_report`/matriz de confusão com **corte implícito 0,5** — informativo, porém secundário. Mais grave: chama Log Loss baixo de "**excelente calibração**" **sem nenhuma curva de calibração**; e `RandomForest` com `class_weight='balanced'` tende a produzir probabilidades distorcidas. A boa Log Loss aqui também é artefato do vazamento.

---

## 8. Entregáveis

- **Coerência notebook × README × submissão:** 🟠
  - README cita `notebook.ipynb` (inexistente) — ver §1.
  - **`submissao_case.csv` atual NÃO foi gerado pelo notebook original.** 1ª linha da submissão = `0,0445`; a saída salva do notebook **original** para a mesma linha = `0,0057`; já a versão de trabalho `case_datarisk.ipynb` produz `0,0445`. Ou seja, o CSV entregue corresponde ao **notebook novo**, não à "solução anterior". Incoerência de proveniência.
  - Formato da submissão em si: ✅ 3 colunas exatas (`ID_CLIENTE, SAFRA_REF, PROBABILIDADE_INADIMPLENCIA`), 12 275 linhas, probabilidades em [0,0007; 0,671] ⊂ [0,1], **ordem do teste preservada** (sequência de `ID_CLIENTE` idêntica à base de teste).
- **Reprodutibilidade:** 🔴 **o notebook original NÃO roda de ponta a ponta no ambiente instalado.** A reexecução em kernel limpo **falha na célula [140]** com `TypeError: RocCurveDisplay.from_predictions() got an unexpected keyword argument 'color'` (o argumento deixou de ser aceito no sklearn ≥1.9). A falha ocorre **antes** da CV/tuning/submissão — portanto, no ambiente atual, a submissão nem chega a ser gerada por ele.
- **Dependências:** 🔴/🟠 `requirements.txt` fixa `scikit‑learn==1.6.1` (onde o `color` provavelmente funcionava), mas o venv tem **1.9.0** (e pandas 3.0.5, numpy 2.5.1) — **ambiente ≠ requirements**. Ou se recria o ambiente do `requirements.txt`, ou se ajusta o código para a versão instalada. `requirements-original.txt` (freeze UTF‑16) só adiciona confusão.
- **Informações pessoais:** ✅ `README/AGENTS/CHECKLIST/requirements` e o texto dos notebooks estão limpos (a ocorrência "github.com" vem do repr HTML de objetos sklearn, não é dado pessoal). 🟢 **Atenção:** o autor do Git é "Ricardo" (metadados do repositório) e há `AGENTS.md`/`CHECKLIST.md` internos — devem ficar **fora do .zip** para respeitar o anonimato exigido pelo PDF.
- **Conformidade com o PDF:** target (§3), colunas/ordem da submissão e uso exclusivo do dev para treino estão conformes; **não‑conformidades**: reprodutibilidade (versões), e a exigência de "aproveitar o histórico de comportamento" pouco explorada (§6).

---

## 9. Células Markdown

- **Erros de português:** 🟢 [15] "A tabela **a abaixo**" (duplicação); [53]/[54] lista `` `DDD` `RENDA_MES_ANTERIOR` `` sem vírgula; [69] "assimétricas **e contendo** outliers" (concordância). Vários acentos/pontuação menores.
- **Textos incompletos / genéricos:** 🟢 muitos comentários de código apenas descritivos ("# Exibe o head…", "# Verifica…") sem interpretar resultado; markdowns de "Introdução/Conclusão" repetem o óbvio.
- **Interpretações incorretas:** 🔴 [123] "Random Forest … **mais robusto e confiável para produção**" e "**excelente calibração**"; [133] RF com "**ótima precisão e recall** … para inadimplentes" — o **recall da classe 1 do RF é 0,652**, enquanto o **XGBoost tem 0,857**. README §5.2 repete "**melhor desempenho na classe minoritária**" para o RF — **contradiz o próprio `classification_report`**.
- **Conclusões incompatíveis com as saídas:** 🔴 [149] "estabilidade confirmada via CV" (CV vazada); todo o discurso de superioridade do RF assenta sobre o AUC inflado (§7).
- **Afirmações fortes demais:** 🔴 "confiável para produção", "excelente/ótima", "garante generalização" — violam também o `AGENTS.md` (proibido alegar prontidão para produção/generalização garantida).
- **Números escritos manualmente:** 🟠 README §5.1 traz métricas "~0,81 / ~0,94 / ~0,96" e "AUC ≈ 0,9624, Log Loss ≈ 0,1170" **fora** do notebook (risco de dessincronizar com a execução); a tabela de resultados no notebook ([122]) faz `sort_values(...)` **sem reatribuir** e depois imprime `results` não‑ordenado — a ordenação "some".
- **Células que descrevem código sem interpretar resultado:** 🟢 [9], [11]–[14], [79], [88], [91], [164] etc.

---

## Tabela de Achados (arquivo · local · problema · correção mínima · impacto)

| Classe | Arquivo · Local | Problema | Correção mínima recomendada | Impacto esperado |
|---|---|---|---|---|
| 🔴 | `case_datarisk_notebook_original.ipynb` · cél. [98] | Split **aleatório por linha** vaza cliente (AUC 0,96 vs 0,78 por cliente / 0,91 temporal) | Adotar **holdout temporal** (validar em safras mais recentes) e/ou `GroupKFold`/`GroupShuffleSplit` por `ID_CLIENTE` | Métrica passa a refletir o desempenho real; corrige a base de toda a seleção de modelo |
| 🔴 | idem · cél. [148]/[151] | CV e tuning usam KFold aleatório → "estabilidade" e ganho ilusórios | Usar `StratifiedGroupKFold` por cliente ou CV temporal | Elimina falsa confiança; tuning passa a otimizar métrica honesta |
| 🔴 | idem · cél. [140]; `requirements.txt` | Não executa ponta‑a‑ponta (sklearn ≥1.9 rejeita `color=`); ambiente ≠ requirements | Remover `color=` (ou usar `ax`/`plot_chance_level`) **ou** fixar o ambiente em sklearn 1.6.1 | Restaura reprodutibilidade exigida pelo PDF |
| 🔴 | `README.md` §5.2 · nb [123]/[133] | Alega RF melhor na classe minoritária e "produção"/"calibração" contra as saídas | Corrigir texto: XGB tem maior recall(1); remover claims de produção/calibração; incluir curva de calibração se quiser falar de calibração | Coerência entre narrativa e evidência; conformidade com `AGENTS.md` |
| 🟠 | nb cél. [77]/[78] | `TAXA_RELATIVA` redundante (corr 0,94) e mal nomeada | Remover, ou trocar por juros/renda; renomear para `CUSTO_JUROS` | Modelo mais enxuto; evita erro conceitual em apresentação |
| 🟠 | nb cél. [88] | `DIAS_ATRASO`/`DATA_PAGAMENTO` permanecem em `X` | `X = dev_full.drop(columns=['TARGET','DIAS_ATRASO','DATA_PAGAMENTO'])` | Remove risco latente de vazamento do alvo |
| 🟠 | nb cél. [75] | Imputação fora do pipeline e mediana no dev inteiro (pré‑split) | Mover imputação para dentro do `Pipeline`/`ColumnTransformer` (`SimpleImputer`) | Elimina vazamento de estatística e garante CV correta |
| 🟠 | `README.md` §1/§2 | Referência a `notebook.ipynb` inexistente | Alinhar nome ao arquivo real da entrega | Evita entrega quebrada para o avaliador |
| 🟠 | `submissao_case.csv` | Gerado por notebook diferente do original | Regerar a submissão a partir do notebook de entrega definitivo | Proveniência coerente |
| 🟠 | `data/*` · `DATA_VENCIMENTO` | Datas até 2027 (18 reg.) não tratadas; `VALOR_A_PAGAR` com missing | Sinalizar/limitar vencimentos absurdos; documentar imputação de `VALOR_A_PAGAR` | `PRAZO_EMISSAO_VENC` mais confiável |
| 🟠 | nb §7 (EDA) | Falta EDA temporal e por cliente; pairplot redundante | Adicionar taxa de default por safra e por cliente; remover pairplot | Revela a estrutura que causa o vazamento; EDA mais objetiva |
| 🟢 | `requirements-original.txt` | Freeze UTF‑16 completo, confuso | Remover da entrega | Menos ruído |
| 🟢 | nb cél. [29] | Comentário "1 se houve atraso" impreciso | Ajustar para "atraso ≥ 5 dias" | Clareza |
| 🟢 | nb cél. [122] | `sort_values` sem reatribuição | `results = results.sort_values(...)` | Saída ordenada como o texto sugere |
| 🟢 | `AGENTS.md`/`CHECKLIST.md`, autor Git | Artefatos internos / nome no histórico | Não incluir no .zip final | Respeita anonimato do PDF |
| 🟢 | Markdown geral | Typos e comentários genéricos | Revisão ortográfica; interpretar resultados em vez de descrever código | Legibilidade/organização (critério avaliado) |

---

### Nota de método
Todos os números empíricos acima foram reproduzidos a partir dos CSVs em `data/` com o venv do projeto. O AUC do split aleatório (0,9624) bateu exatamente com a saída salva do notebook, validando a comparação com os splits por cliente (0,7828) e temporal (0,9063). Nenhum arquivo existente foi alterado; apenas este `BASELINE_AUDIT.md` foi criado.
