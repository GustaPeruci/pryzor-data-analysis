# Relatório de Processamento de Dados - Pryzor

## Análise de Melhor Momento para Compra de Jogos na Steam

---

## 1. RESUMO EXECUTIVO

Este relatório documenta o processamento completo de dados para um projeto de portifólio de projeto utilizando IA para determinar o melhor momento para compra de jogos na Steam. O projeto seguiu rigorosamente as 5 etapas metodológicas propostas:

**Resultados Principais:**
- ✅ **2.000 jogos** analisados no dataset principal
- ✅ **100 jogos** com dados completos de histórico de preços
- ✅ **24 features** criadas através de engenharia de recursos
- ✅ **Datasets** preparados para treino (70%), validação (10%) e teste (20%)
- ✅ **Variável target** criada para classificação de "bom momento para compra"

---

## 2. ESTRUTURA DOS DADOS ORIGINAIS

### 2.1 Dataset Principal (`applicationInformation.csv`)
- **Registros:** 2.000 jogos
- **Colunas:** 5 (appid, type, name, releasedate, freetoplay)
- **Tipos de aplicações:**
  - Games: 1.851 (92.6%)
  - Demos: 13 (0.7%)
  - Advertising: 12 (0.6%)
  - Mods: 8 (0.4%)
  - DLC: 3 (0.2%)

### 2.2 Histórico de Preços (`PriceHistory/*.csv`)
- **Formato:** Um arquivo CSV por jogo (nome = app_id)
- **Período:** Abril 2019 - Agosto 2020
- **Colunas:** Date, Initialprice, Finalprice, Discount
- **Frequência:** Dados diários

---

## 3. ETAPA 1: EXPLORAÇÃO DE ESTATÍSTICAS DESCRITIVAS

### 3.1 Análise do Dataset Principal
```
Dimensões: 2.000 x 5
Valores ausentes identificados:
- type: 113 registros (5.7%)
- releasedate: 139 registros (7.0%)
- freetoplay: 121 registros (6.1%)
```

### 3.2 Análise de Preços
- **Jogos gratuitos:** 372 (19.8%)
- **Jogos pagos:** 1.507 (80.2%)
- **Período de observação:** ~495 dias por jogo
- **Range de preços:** $0.69 - $24.99

### 3.3 Insights Iniciais
- Distribuição de preços concentrada em faixas baixas ($9.99 - $19.99)
- Presença significativa de jogos gratuitos (F2P)
- Dados temporais consistentes entre 2019-2020

---

## 4. ETAPA 2: LIMPEZA DE DADOS

### 4.1 Tratamento de Valores Ausentes
- **type:** Preenchido com "unknown"
- **releasedate:** Preenchido com "Unknown"
- **freetoplay:** Preenchido com 0 (assumindo jogo pago)

### 4.2 Qualidade dos Dados
- **Duplicatas removidas:** Baseado no app_id
- **Filtro de qualidade:** Mantidos apenas jogos com >10 registros de preço
- **Resultado final:** 100 jogos com dados completos e confiáveis

### 4.3 Problemas Identificados e Soluções
- **Encoding:** Resolvido com latin-1
- **Consistência temporal:** Dados alinhados por data
- **Outliers:** Mantidos para preservar variações reais de preço

---

## 5. ETAPA 3: TRANSFORMAÇÃO DE VARIÁVEIS

### 5.1 Transformações Temporais
- **Conversão de datas:** String → DateTime
- **Extração de features temporais:**
  - Ano, mês, dia da semana
  - Quarter (trimestre)
  - Indicadores sazonais

### 5.2 Transformações de Preço
- **Economia absoluta:** Initialprice - Finalprice
- **Indicador de desconto:** Binário (0/1)
- **Percentual de desconto:** Normalizado

### 5.3 Codificação de Variáveis Categóricas
- **Tipo de jogo:** Label encoding
- **Ano de lançamento:** Extraído de releasedate

---

## 6. ETAPA 4: ENGENHARIA DE RECURSOS

### 6.1 Features Criadas (24 total)

#### Características do Jogo:
1. `app_id` - Identificador único
2. `game_name` - Nome do jogo
3. `game_type` - Tipo da aplicação
4. `is_free_to_play` - Indicador F2P
5. `release_year` - Ano de lançamento

#### Estatísticas de Preço:
6. `avg_initial_price` - Preço inicial médio
7. `min_initial_price` - Preço inicial mínimo
8. `max_initial_price` - Preço inicial máximo
9. `price_volatility` - Desvio padrão dos preços
10. `avg_final_price` - Preço final médio
11. `min_final_price` - Preço final mínimo
12. `max_final_price` - Preço final máximo

#### Análise de Descontos:
13. `avg_discount` - Desconto médio (%)
14. `max_discount` - Desconto máximo (%)
15. `discount_frequency` - Frequência de descontos (0-1)
16. `total_discount_days` - Total de dias com desconto

#### Features Temporais:
17. `total_observation_days` - Período total observado
18. `price_trend` - Tendência de preço (correlação temporal)
19. `seasonal_discount_pattern` - Variância sazonal de descontos

#### Indicadores de Oportunidade:
20. `best_discount_month` - Mês com melhores descontos
21. `worst_discount_month` - Mês com piores descontos
22. `max_savings` - Economia máxima possível ($)
23. `avg_savings_when_discounted` - Economia média em promoções

#### Variável Target:
24. `good_buy_time` - **Target binário:** Bom momento para compra (0/1)

### 6.2 Critérios para Target Variable
Um jogo é considerado "bom momento para compra" quando atende ≥2 critérios:
- ✓ Frequência de desconto > mediana
- ✓ Desconto médio > mediana  
- ✓ Economia máxima > mediana

---

## 7. ETAPA 5: DIVISÃO DE DADOS

### 7.1 Estratégia de Divisão
```
Dataset Total: 100 jogos
├── Treino: 70 jogos (70%)
├── Validação: 10 jogos (10%)
└── Teste: 20 jogos (20%)
```

### 7.2 Normalização
- **Método:** StandardScaler (μ=0, σ=1)
- **Aplicação:** Apenas features numéricas
- **Preservação:** Scaler salvo para inferência futura

### 7.3 Balanceamento
- **Estratificação:** Aplicada na variável target
- **Distribuição preservada** entre train/val/test

---

## 8. INSIGHTS E DESCOBERTAS

### 8.1 Padrões de Preço Identificados
- **Sazonalidade:** Dezembro apresenta os melhores descontos
- **Volatilidade:** Jogos mais caros tendem a ter maior volatilidade
- **Tendência:** Maioria dos jogos mantém preços estáveis ao longo do tempo

### 8.2 Comportamento de Desconto
- **Frequência média:** ~13% dos dias com desconto
- **Desconto médio:** 7-10% quando em promoção
- **Picos sazonais:** Dezembro e fevereiro

### 8.3 Características dos "Bons Momentos"
- Jogos com **alta frequência de desconto** (>13%)
- **Descontos significativos** (>8%)
- **Economia substancial** (>$7)

---

## 9. ARQUIVOS GERADOS

### 9.1 Dados Processados
- `processed_application_info.csv` - Dataset principal limpo
- `engineered_features.csv` - Features engineered completas
- `steam_games_analysis.png` - Visualizações exploratórias

### 9.2 Metadados
- `processing_summary.txt` - Resumo quantitativo
- `steam_data_processing.py` - Script completo documentado

---

## 10. PRÓXIMOS PASSOS PARA MODELAGEM

### 10.1 Modelos Sugeridos
1. **Random Forest** - Para importância de features
2. **XGBoost** - Para performance superior
3. **Logistic Regression** - Para interpretabilidade
4. **Neural Networks** - Para capturar padrões complexos

### 10.2 Métricas de Avaliação
- **Acurácia** - Performance geral
- **Precision/Recall** - Para classe "bom momento"
- **F1-Score** - Balanço precision/recall
- **AUC-ROC** - Capacidade discriminativa

### 10.3 Validação
- **Cross-validation** nos dados de treino
- **Validação temporal** (se aplicável)
- **Teste final** no conjunto reservado

---

## 11. CONCLUSÕES

### ✅ Objetivos Alcançados
1. **Exploração completa** dos dados brutos
2. **Limpeza robusta** com documentação de decisões
3. **Transformações adequadas** para análise temporal
4. **Features relevantes** para o problema de negócio
5. **Divisão apropriada** para modelagem ML
