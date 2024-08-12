import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import uniform, randint
import concurrent.futures
from sklearn.linear_model import ElasticNet


def obtener_cotizacion_dolar():
    url = "https://dolarhoy.com/cotizacion-dolar-blue"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        div_value = soup.find('div', class_='value').text
        cotizacion_dolar = float(div_value.replace('$', '').replace(',', '.'))
        return cotizacion_dolar
    except Exception as e:
        print(f"Error: {e}")
        return None

cotizacion_dolar = obtener_cotizacion_dolar()

print(f"Cotización del dólar: {cotizacion_dolar}")

def construct_search_url(base_url, year_start, year_end, brand, model, version):
    query = f"{year_start}-{year_end}/{brand}-{model}-{version.replace(' ', '-')}"
    search_url = f"{base_url}{query}"
    return search_url

base_url = 'https://autos.mercadolibre.com.ar/'
year_start = 2012
year_end = 2024
brand = 'Toyota'
model = "etios"
version = "xls"
user_year = 2017
user_km = 30000
search_url = construct_search_url(base_url, year_start, year_end, brand, model, version)
print(f"URL de búsqueda: {search_url}")

def fetch_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup

def parse_listings(soup, cotizacion_dolar):
    listings = []
    current_listings = soup.find_all('li', class_='ui-search-layout__item')
    for listing in current_listings:
        price_tag = listing.find('span', class_='andes-money-amount__fraction')
        price_currency_tag = listing.find('span', class_='andes-money-amount__currency-symbol')
        year_tag = listing.find('li', class_='ui-search-card-attributes__attribute')
        km_tag = listing.find_all('li', class_='ui-search-card-attributes__attribute')[1]
        if price_tag and year_tag and km_tag:
            price = int(price_tag.get_text().replace('.', ''))
            currency = price_currency_tag.get_text().strip() if price_currency_tag else '$'
            if currency == 'US$':
                price *= cotizacion_dolar
            year = int(year_tag.get_text().strip())
            km = int(km_tag.get_text().replace(' Km', '').replace('.', '').strip())
            listings.append({'price': price, 'year': year, 'km': km})
    return listings

def scrape_mercadolibre(search_url, cotizacion_dolar):
    listings = []
    page = 1
    max_workers = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        while True:
            url = f"{search_url}_Desde_{(page - 1) * 48 + 1}"
            futures.append(executor.submit(fetch_page, url))
            page += 1
            if len(futures) >= max_workers:
                results = [future.result() for future in futures]
                if all(len(result.find_all('li', class_='ui-search-layout__item')) == 0 for result in results):
                    break
                for result in results:
                    listings.extend(parse_listings(result, cotizacion_dolar))
                futures = []

        results = [future.result() for future in futures]
        for result in results:
            listings.extend(parse_listings(result, cotizacion_dolar))

    df = pd.DataFrame(listings)
    return df

start_time = time.time()
listings = scrape_mercadolibre(search_url, cotizacion_dolar)
end_time = time.time()
print(f"Tiempo de ejecución del scraping: {end_time - start_time} segundos")

print(listings)

def remove_outliers_and_zero_km(df, columns):
    df = df[df['km'] != 0]
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

start_time = time.time()
df_clean = remove_outliers_and_zero_km(listings, ['price', 'year', 'km'])
end_time = time.time()
print(f"Tiempo de ejecución de la limpieza de datos: {end_time - start_time} segundos")

print("DataFrame después de eliminar outliers y kilometraje igual a 0:")
print(df_clean)

print("Estadísticas descriptivas:")
print(df_clean.describe())

# Gráficos de distribución de variables
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
sns.histplot(df_clean['price'], bins=10, kde=True)
plt.title('Distribución de Precios')

plt.subplot(1, 3, 2)
sns.histplot(df_clean['year'], bins=10, kde=True)
plt.title('Distribución de Años')

plt.subplot(1, 3, 3)
sns.histplot(df_clean['km'], bins=10, kde=True)
plt.title('Distribución de Kilometraje')
plt.show()

# Relación entre variables
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x='km', y='price', data=df_clean)
plt.title('Relación Precio vs Kilometraje')

plt.subplot(1, 2, 2)
sns.scatterplot(x='year', y='price', data=df_clean)
plt.title('Relación Precio vs Año')
plt.show()

# Correlación entre variables
print("Correlación entre variables:")
print(df_clean.corr())

# Gráficos adicionales
plt.figure(figsize=(10, 6))
sns.boxplot(x='year', y='price', data=df_clean)
plt.title('Distribución de Precios por Año')
plt.show()

df_clean['km_bin'] = pd.cut(df_clean['km'], bins=[0, 20000, 40000, 60000, 80000, 100000, 120000, 140000], labels=['0-20k', '20k-40k', '40k-60k', '60k-80k', '80k-100k', '100k-120k', '120k-140k'])
plt.figure(figsize=(10, 6))
sns.boxplot(x='km_bin', y='price', data=df_clean)
plt.title('Distribución de Precios por Rangos de Kilometraje')
plt.show()

# Variables explicativas y variable objetivo
start_time = time.time()
X = df_clean[['year', 'km']]
y = df_clean['price']
end_time = time.time()
print(f"Tiempo de ejecución de la asignación de variables X e y: {end_time - start_time} segundos")

# Dividir el dataset en entrenamiento y prueba
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
end_time = time.time()
print(f"Tiempo de ejecución de la división del dataset: {end_time - start_time} segundos")

# Definir parámetros para RandomizedSearchCV
param_dist_svr = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': uniform(0.1, 10),
    'epsilon': uniform(0.1, 0.2)
}

param_dist_knn = {
    'n_neighbors': randint(3, 10),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

param_dist_xgb = {
    'n_estimators': randint(50, 100),
    'max_depth': randint(3, 6),
    'learning_rate': uniform(0.01, 0.1),
    'subsample': uniform(0.7, 0.3)
}

param_dist_en = {
    'alpha': uniform(0.01, 10),
    'l1_ratio': uniform(0, 1)
}

# Crear modelos para RandomizedSearchCV
svr = SVR()
knn = KNeighborsRegressor()
xgb = XGBRegressor(random_state=42)
en = ElasticNet(random_state=42)

# Crear RandomizedSearchCV con menos iteraciones y pliegues
random_search_svr = RandomizedSearchCV(svr, param_distributions=param_dist_svr, n_iter=10, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
random_search_knn = RandomizedSearchCV(knn, param_distributions=param_dist_knn, n_iter=10, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
random_search_xgb = RandomizedSearchCV(xgb, param_distributions=param_dist_xgb, n_iter=10, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
random_search_en = RandomizedSearchCV(en, param_distributions=param_dist_en, n_iter=10, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
# Entrenar modelos con RandomizedSearchCV
start_time = time.time()
try:
    random_search_svr.fit(X_train, y_train)
    print(f"Best SVR params: {random_search_svr.best_params_}")
except Exception as e:
    print(f"Error during SVR fitting: {e}")
end_time = time.time()
print(f"Tiempo de entrenamiento de SVR: {end_time - start_time} segundos")

start_time = time.time()
try:
    random_search_knn.fit(X_train, y_train)
    print(f"Best KNN params: {random_search_knn.best_params_}")
except Exception as e:
    print(f"Error during KNN fitting: {e}")
end_time = time.time()
print(f"Tiempo de entrenamiento de KNN: {end_time - start_time} segundos")

start_time = time.time()
try:
    random_search_xgb.fit(X_train, y_train)
    print(f"Best XGB params: {random_search_xgb.best_params_}")
except Exception as e:
    print(f"Error during XGB fitting: {e}")
end_time = time.time()
print(f"Tiempo de entrenamiento de XGB: {end_time - start_time} segundos")

# Entrenar el modelo con RandomizedSearchCV
start_time = time.time()
try:
    random_search_en.fit(X_train, y_train)
    print(f"Best ElasticNet params: {random_search_en.best_params_}")
except Exception as e:
    print(f"Error during ElasticNet fitting: {e}")
end_time = time.time()
print(f"Tiempo de entrenamiento de ElasticNet: {end_time - start_time} segundos")

# Obtener los mejores modelos
best_svr = random_search_svr.best_estimator_
best_knn = random_search_knn.best_estimator_
best_xgb = random_search_xgb.best_estimator_
best_en = random_search_en.best_estimator_

# Evaluar los mejores modelos
def evaluate_model(model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()
    print(f"Tiempo de evaluación del modelo {model.__class__.__name__}: {end_time - start_time} segundos")
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, mae, r2, y_pred

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Support Vector Regression': best_svr,
    'K-Nearest Neighbors': best_knn,
    'XGBoost': best_xgb,
    'Elastic Net': best_en
}

# Evaluar cada modelo y seleccionar los mejores
# Evaluar cada modelo y seleccionar los mejores
best_model = None
best_score = float('inf')
results = {}
model_predictions = {}

start_time = time.time()
for name, model in models.items():
    mse, mae, r2, y_pred = evaluate_model(model, X_train, y_train, X_test, y_test)
    results[name] = {'MSE': mse, 'MAE': mae, 'R²': r2}
    if mse < best_score:
        best_score = mse
        best_model = model
        best_model_name = name
    model_predictions[name] = y_pred
end_time = time.time()
print(f"Tiempo de evaluación de todos los modelos: {end_time - start_time} segundos")

# Mostrar resultados
for name, metrics in results.items():
    print(f"{name}:\n  MSE: {metrics['MSE']}\n  MAE: {metrics['MAE']}\n  R²: {metrics['R²']}\n")

# Seleccionar los 3 mejores modelos con R² positivo, o los 3 mejores aunque sean negativos
sorted_models = sorted(results.items(), key=lambda x: x[1]['R²'], reverse=True)
top_models = [model for model in sorted_models if model[1]['R²'] > 0]

if len(top_models) < 3:
    top_models = sorted_models[:3]

selected_models = [models[name] for name, _ in top_models]

# Crear un ensemble de los mejores modelos seleccionados
ensemble = VotingRegressor(estimators=[(name, model) for name, model in zip([name for name, _ in top_models], selected_models)])
ensemble.fit(X_train, y_train)

# Evaluar el modelo de ensamble
mse_ensemble, mae_ensemble, r2_ensemble, y_pred_ensemble = evaluate_model(ensemble, X_train, y_train, X_test, y_test)
results['Ensemble'] = {'MSE': mse_ensemble, 'MAE': mae_ensemble, 'R²': r2_ensemble}
model_predictions['Ensemble'] = y_pred_ensemble

# Entrenar el mejor modelo con todos los datos
start_time = time.time()
best_model.fit(X, y)
end_time = time.time()
print(f"Tiempo de entrenamiento del mejor modelo con todos los datos: {end_time - start_time} segundos")

# Añadir el modelo de ensamble a la función de predicción
models['Ensemble'] = ensemble

def predict_price(year, km, models, cotizacion_dolar):
    data = pd.DataFrame({'year': [year], 'km': [km]})
    predictions = {}
    for name, model in models.items():
        price_pesos = model.predict(data)[0]
        price_dolares = price_pesos / cotizacion_dolar
        predictions[name] = {'pesos': price_pesos, 'dolares': price_dolares}
    return predictions


start_time = time.time()
predictions = predict_price(user_year, user_km, models, cotizacion_dolar)
end_time = time.time()
print(f"Tiempo de ejecución de predict_price: {end_time - start_time} segundos")

# Mostrar el mejor modelo y las predicciones
print(f"El mejor modelo es: {best_model_name}\n")
print(f"Predicciones de precios para su vehículo (Año: {user_year}, Km: {user_km}):")
for model_name, prices in predictions.items():
    print(f"{model_name}: ${prices['pesos']:.2f} pesos / ${prices['dolares']:.2f} dólares")

# Graficar resultados
plt.figure(figsize=(15, 10))

# Gráfico 1: Comparación de predicciones de los modelos con los valores reales
plt.subplot(2, 2, 1)
top_model_names = [name for name, _ in sorted_models[:3]] + ['Ensemble']
for name, y_pred in model_predictions.items():
    if name in top_model_names:
        sns.lineplot(x=y_test, y=y_pred, label=name)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Precio Real')
plt.ylabel('Precio Predicho')
plt.title('Comparación de Predicciones de los Modelos')
plt.legend()

# Gráfico 2: Distribución de precios predichos
plt.subplot(2, 2, 2)
predictions_df = pd.DataFrame(model_predictions)
predictions_df['Real'] = y_test.values
sns.boxplot(data=predictions_df)
plt.xticks(rotation=45)
plt.title('Distribución de Precios Predichos')

# Gráfico 3: Relación entre Year y Precio
plt.subplot(2, 2, 3)
sns.lineplot(x=X_test['year'], y=y_test, label='Real', color='blue')
for name in top_model_names:
    sns.lineplot(x=X_test['year'], y=model_predictions[name], label=name)
plt.xlabel('Año')
plt.ylabel('Precio')
plt.title('Relación entre Año y Precio')
plt.legend()

# Gráfico 4: Relación entre Km y Precio
plt.subplot(2, 2, 4)
sns.lineplot(x=X_test['km'], y=y_test, label='Real', color='blue')
for name in top_model_names:
    sns.lineplot(x=X_test['km'], y=model_predictions[name], label=name)
plt.xlabel('Kilometraje')
plt.ylabel('Precio')
plt.title('Relación entre Kilometraje y Precio')
plt.legend()

plt.tight_layout()
plt.show()

# Conclusión Final
print("\nConclusión Final:\n")
print("El análisis ha permitido identificar los tres mejores modelos para la predicción de precios. A continuación, se presentan las métricas y las predicciones de estos modelos para el vehículo con Año: 2014 y Kilometraje: 45000.")

# Mostrar los tres mejores modelos y sus métricas
sorted_results = sorted(results.items(), key=lambda x: x[1]['R²'], reverse=True)
for name, metrics in sorted_results[:3]:
    print(f"{name}:\n  MSE: {metrics['MSE']}\n  MAE: {metrics['MAE']}\n  R²: {metrics['R²']}\n")

# Mostrar las predicciones finales
print("\nPredicciones de precios para el vehículo:")
for model_name, prices in predictions.items():
    if model_name in [name for name, _ in sorted_results[:3]] or model_name == 'Ensemble':
        print(f"{model_name}: ${prices['pesos']:.2f} pesos / ${prices['dolares']:.2f} dólares")

# Si el modelo de ensamble es uno de los tres mejores, inclúyelo
if 'Ensemble' in [name for name, _ in sorted_results[:3]]:
    ensemble_price = predictions['Ensemble']
    print(f"Ensemble: ${ensemble_price['pesos']:.2f} pesos / ${ensemble_price['dolares']:.2f} dólares")

print("\nConclusión:")
print(f"Se han utilizado {len(X)} autos usados para el análisis.")
print("La comparación de los modelos muestra que el modelo de regresión lineal es el mejor en términos generales. Sin embargo, los modelos de ensamble y los tres mejores modelos ofrecen predicciones competitivas.")
print("Es recomendable considerar los modelos de ensamble si se busca una mayor precisión en la predicción. Las predicciones presentadas son las más relevantes para el cliente, proporcionando una visión clara de los precios estimados para el vehículo en cuestión.")
