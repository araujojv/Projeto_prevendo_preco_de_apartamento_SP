# Previsão de Preços de Aluguel em São Paulo

## Descrição do Projeto

Este projeto tem como objetivo analisar e prever os preços de aluguel de apartamentos na cidade de São Paulo. Utilizamos um dataset contendo informações sobre as propriedades, incluindo localização geográfica, tamanho, e outras características relevantes para determinar o preço de aluguel.

## Tecnologias Utilizadas
- **Linguagem**: Python
- **Bibliotecas**:
  - `pandas`: manipulação de dados
  - `numpy`: operações matemáticas
  - `matplotlib` e `seaborn`: visualização de dados
  - `plotly`: criação de mapas interativos
  - `scikit-learn`: modelagem preditiva
  - `geopy`: cálculo de distância geográfica
  - `xgboost`: modelo de aprendizado de máquina

## Etapas do Projeto

### 1. Importação dos Dados
Os dados foram carregados a partir de um arquivo CSV:
```python
import pandas as pd

# Carregar os dados
file_path = "sao-paulo-properties-april-2019.csv"
df_data = pd.read_csv(file_path)
```

### 2. Filtragem de Dados
Selecionamos apenas os imóveis que são destinados para aluguel:
```python
df_rent = df_data[df_data["Negotiation Type"] == "rent"]
```

### 3. Visualização Geoespacial
Criamos um mapa interativo utilizando Plotly para visualizar a localização e os preços dos apartamentos:
```python
import plotly.express as px
import plotly.graph_objects as go

fig = px.scatter_mapbox(df_rent, lat="Latitude", lon="Longitude", color="Price", size="Size",
                        color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10, opacity=0.4)
fig.show()
```

### 4. Análise Estatística
- Distribuição das variáveis:
```python
df_rent.hist(bins=30, figsize=(30, 15))
```
- Contagem de tipos de propriedades:
```python
df_rent["Property Type"].value_counts()
```
- Correlação entre variáveis e o preço:
```python
df_rent.select_dtypes(include=['number']).corr()["Price"].sort_values(ascending=False)
```

### 5. Processamento dos Dados
Removemos colunas irrelevantes para o treinamento do modelo:
```python
df_cleaned = df_rent.drop(["New", "Property Type", "Negotiation Type"], axis=1)
```

Adicionamos uma coluna calculando a distância do imóvel até o centro de São Paulo:
```python
from geopy.distance import geodesic

centro_sp = (-23.5505, -46.6333)
df_cleaned["Distancia_centro"] = df_cleaned.apply(lambda row: geodesic(centro_sp, (row["Latitude"], row["Longitude"])).km, axis=1)
```

### 6. Modelagem Preditiva

#### Separando os dados de treino e teste:
```python
from sklearn.model_selection import train_test_split

Y = df_cleaned["Price"] ## Variável alvo
X = df_cleaned.drop(columns=["Price"])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
```

#### Modelos Utilizados:
Foram testados os seguintes modelos de Machine Learning:
- **Regressão Linear**
- **Árvore de Decisão**
- **Random Forest**
- **XGBoost**

Treinamento e avaliação:
```python
from sklearn.metrics import mean_squared_error, r2_score

modelos = {
    "Regressão Linear": LinearRegression(),
    "Árvore de Decisão": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

resultados = {}
for nome, modelo in modelos.items():
    modelo.fit(x_train, y_train)
    preds = modelo.predict(x_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    resultados[nome] = {"MSE": mse, "R²": r2}
```

### 7. Resultado Final
O modelo **Random Forest** apresentou o melhor desempenho, sendo selecionado como modelo final:
```python
best_rf_model = RandomForestRegressor()
best_rf_model.fit(x_train, y_train)
```

## Como Executar o Projeto
1. Instale as dependências necessárias:
   ```bash
   pip install pandas numpy matplotlib seaborn plotly scikit-learn geopy xgboost
   ```
2. Execute o notebook Jupyter.

## Contribuição
Se desejar contribuir com melhorias, abra uma issue ou faça um pull request!

## Licença
Este projeto está sob a licença MIT.

