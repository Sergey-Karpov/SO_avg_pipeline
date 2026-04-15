import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import joblib
import warnings

warnings.filterwarnings('ignore')


# ==================== КАСТОМНЫЕ ТРАНСФОРМЕРЫ ====================

class OutlierHandler(BaseEstimator, TransformerMixin):
    """Обработка выбросов в колонке 'avg'"""

    def __init__(self, column='avg'):
        self.column = column
        self.lower_bound = None
        self.upper_bound = None

    def fit(self, X, y=None):
        q1 = X[self.column].quantile(0.25)
        q3 = X[self.column].quantile(0.75)
        iqr = q3 - q1
        self.lower_bound = q1 - 1.5 * iqr
        self.upper_bound = q3 + 1.5 * iqr
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy[self.column] = X_copy[self.column].clip(self.lower_bound, self.upper_bound)
        return X_copy


class FeatureCreator(BaseEstimator, TransformerMixin):
    """Создание новых признаков из cereals и milk"""
    def __init__(self, use_predict=False):
        self.use_predict = use_predict
        self.store_counts = None

    def fit(self, X, y=None):
        if not self.use_predict:
            self.store_counts = pd.crosstab(X["city"], X["chain"])
            return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy["cereals_milk_ratio"] = X_copy["cereals"] / (X_copy["milk"] + 1)
        X_copy["cereals_milk_multi"] = X_copy["cereals"] * X_copy["milk"]
        if not self.use_predict:
            X_copy["aushan_count_in_city"] = X_copy["city"].map(self.store_counts["Ашан"]).fillna(0)
            X_copy["detmir_count_in_city"] = X_copy["city"].map(self.store_counts["Детский мир"]).fillna(0)
            X_copy["lenta_count_in_city"] = X_copy["city"].map(self.store_counts["Лента"]).fillna(0)

        else:
            required_cols = ["aushan_count_in_city", "detmir_count_in_city", "lenta_count_in_city"]
            missing_cols = [col for col in required_cols if col not in X_copy.columns]

            if missing_cols:
                raise ValueError(f"Для предсказания необходимы колонки {missing_cols}")

        return X_copy


class ShareCalculator(BaseEstimator, TransformerMixin):
    """Расчет долей топовых сетей на основе входных данных"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        # Суммируем количество магазинов топовых сетей
        X_copy['top_chains_stores_count'] = (
                X_copy['aushan_count_in_city'] +
                X_copy['detmir_count_in_city'] +
                X_copy['lenta_count_in_city']
        )

        # Избегаем деления на ноль
        denominator = X_copy['top_chains_stores_count'].replace(0, 1)

        # Создаем доли
        X_copy['aushan_count_share_in_city'] = X_copy['aushan_count_in_city'] / denominator
        X_copy['detmir_count_share_in_city'] = X_copy['detmir_count_in_city'] / denominator
        X_copy['lenta_count_share_in_city'] = X_copy['lenta_count_in_city'] / denominator

        return X_copy


class PopulationTransformer(BaseEstimator, TransformerMixin):
    """Добавление данных о населении"""

    def __init__(self, population_file='data/population.xlsx', use_predict=False):
        self.population_file = population_file
        self.use_predict = use_predict
        self.population_data = None
        self.manual_population = {
            'Москва обл': 8775735,
            'Санкт-Петербург обл': 2059479,
            'Симферополь': 335009,
            'Славгород': 27040,
            'Орел': 289503,
            'Артем': 108274
        }

    def fit(self, X, y=None):
        if not self.use_predict:
            try:
                df_pop = pd.read_excel(self.population_file)
                df_pop.rename(columns={
                    'Русское\nназвание': 'city',
                    'Перепись населения 2021 года[3]': 'population'
                }, inplace=True)

                df_pop['population'] = (df_pop['population']
                                        .astype(str)
                                        .str.replace(',', '', regex=False)
                                        .str.replace('.', '', regex=False)
                                        .str.replace(' ', '', regex=False)
                                        .astype(int))

                self.population_data = df_pop
            except:
                print("Не удалось загрузить population.xlsx, использую manual_population")
                self.population_data = None
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        if self.use_predict:
            # Для предсказаний - population уже есть во входных данных
            if 'population' not in X_copy.columns:
                raise ValueError("Для предсказания необходима колонка 'population'")
        else:
            # Для обучения - добавляем population из справочника
            if self.population_data is not None:
                X_copy = X_copy.merge(self.population_data, on='city', how='left')
                X_copy['population'] = X_copy.apply(
                    lambda row: self.manual_population.get(row['city'], row.get('population', 0)),
                    axis=1
                )
            else:
                X_copy['population'] = X_copy['city'].map(self.manual_population).fillna(0)

        return X_copy


class MarketShareTransformer(BaseEstimator, TransformerMixin):
    """Добавление данных о доле рынка"""

    def __init__(self, market_share_file='data/market_share.xlsx', use_predict=False):
        self.market_share_file = market_share_file
        self.use_predict = use_predict
        self.market_share_data = None

    def fit(self, X, y=None):
        if not self.use_predict:
            try:
                df_share = pd.read_excel(self.market_share_file)
                df_share['city'] = df_share['city'].replace({
                    'Московская обл': 'Москва обл',
                    'Ленинградская': 'Санкт-Петербург обл'
                })
                df_share = df_share.rename(columns={'share': 'market_share'})
                self.market_share_data = df_share
            except:
                print("Не удалось загрузить market_share.xlsx")
                self.market_share_data = None
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        if self.use_predict:
            # Для предсказаний - market_share уже есть во входных данных
            if 'market_share' not in X_copy.columns:
                raise ValueError("Для предсказания необходима колонка 'market_share'")
        else:
            # Для обучения - добавляем market_share из справочника
            if self.market_share_data is not None:
                X_copy = X_copy.merge(self.market_share_data, how='left', on='city')
                X_copy['market_share'] = X_copy['market_share'].fillna(0)
            else:
                X_copy['market_share'] = 0

        return X_copy


class InputFeatureValidator(BaseEstimator, TransformerMixin):
    """Валидатор входных признаков для предсказания"""

    def __init__(self, required_features):
        self.required_features = required_features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Проверяем наличие всех обязательных признаков
        missing_features = set(self.required_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Отсутствуют обязательные признаки: {missing_features}")

        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Удаление ненужных колонок"""

    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        columns_to_drop = [col for col in self.columns_to_drop if col in X_copy.columns]
        if columns_to_drop:
            X_copy = X_copy.drop(columns=columns_to_drop, axis=1)
        return X_copy


# ==================== ОСНОВНАЯ ЧАСТЬ (ОБУЧЕНИЕ) ====================

# Загрузка данных
df = pd.read_excel('data/SO.xlsx')
print(f"Исходный размер данных: {df.shape}")

# Создание pipeline для ОБУЧЕНИЯ
training_pipeline = Pipeline([
    ('outlier_handler', OutlierHandler(column='avg')),
    ('feature_creator', FeatureCreator(use_predict=False)),
    ('share_calculator', ShareCalculator()),
    ('population', PopulationTransformer(use_predict=False)),
    ('market_share', MarketShareTransformer(use_predict=False)),
    ('column_dropper', ColumnDropper(['id', 'city']))
])

# Подготовка данных через pipeline
print("Применяем pipeline для подготовки данных...")
df_prepared = training_pipeline.fit_transform(df)
print(f"Размер после pipeline: {df_prepared.shape}")

# Разделение на признаки и целевую переменную
X = df_prepared.drop(['avg'], axis=1)
y = df_prepared['avg']

print(f"\nПризнаки для обучения: {X.columns.tolist()}")

# One-hot encoding для chain
X = pd.get_dummies(X, columns=['chain'], drop_first=True)

print(f"Размерность после one-hot encoding: {X.shape}")
print(f"Колонки после one-hot: {X.columns.tolist()}")

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)

print(f"\nРазмер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Преобразование обратно в DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Создание модели Random Forest
rf_model = RandomForestRegressor(
    n_estimators=600,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=1,
    max_samples=0.65,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
)

# Обучение модели
print("\nОбучение модели...")
rf_model.fit(X_train_scaled, y_train)

# Предсказания и оценка
y_train_pred = rf_model.predict(X_train_scaled)
y_test_pred = rf_model.predict(X_test_scaled)


def evaluate_model(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f'\n---Метрика для {dataset_name} ---')
    print(f'MAE: {mae:.2f}')
    print(f'MSE: {mse:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'R2-score: {r2:.4f}')

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}


train_metrics = evaluate_model(y_train, y_train_pred, 'Обучающей выборки')
test_metrics = evaluate_model(y_test, y_test_pred, 'Тестовой выборки')

# ==================== СОХРАНЕНИЕ МОДЕЛИ ====================

# Определяем ВХОДНЫЕ признаки для предсказания
PREDICTION_INPUT_FEATURES = [
    'chain',  # Сеть магазина
    'cereals',  # Количество круп
    'milk',  # Количество молока
    'population',  # Население города
    'market_share',  # Доля рынка
    'aushan_count_in_city',  # Кол-во магазинов Ашан в городе
    'detmir_count_in_city',  # Кол-во магазинов Детский мир в городе
    'lenta_count_in_city'  # Кол-во магазинов Лента в городе
]

# Создаем pipeline для ПРЕДСКАЗАНИЯ
prediction_pipeline = Pipeline([
    ('input_validator', InputFeatureValidator(PREDICTION_INPUT_FEATURES)),
    ('feature_creator', FeatureCreator(use_predict=True)),
    ('share_calculator', ShareCalculator()),
    ('population', PopulationTransformer(use_predict=True)),
    ('market_share', MarketShareTransformer(use_predict=True)),
    ('column_dropper', ColumnDropper([
        'aushan_count_in_city', 'detmir_count_in_city',
        'lenta_count_in_city', 'top_chains_stores_count'
    ]))
])

# Сохраняем все артефакты
model_artifacts = {
    'prediction_pipeline': prediction_pipeline,
    'scaler': scaler,
    'model': rf_model,
    'feature_columns': X_train.columns.tolist(),
    'input_features': PREDICTION_INPUT_FEATURES,
    'train_metrics': train_metrics,
    'test_metrics': test_metrics,
    'model_params': rf_model.get_params()
}

# Сохранение
joblib.dump(model_artifacts, 'model_artifacts.joblib')
print("\n✅ Модель сохранена в 'model_artifacts.joblib'")

with open('model_artifacts.pkl', 'wb') as file:
    pickle.dump(model_artifacts, file)
print("✅ Модель сохранена в 'model_artifacts.pkl'")

# Анализ важности признаков
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Важность признаков ===")
print(feature_importance.to_string(index=False))


# ==================== ФУНКЦИЯ ДЛЯ ПРЕДСКАЗАНИЯ ====================

def predict_from_input(input_data):
    """
    Функция для предсказания на основе входных данных

    Параметры:
    input_data: pandas DataFrame с колонками (8 признаков, БЕЗ city!):
        - chain: str - сеть магазина
        - cereals: float - количество SKU каш
        - milk: float - количество SKU молока
        - population: int - население города
        - market_share: float - доля рынка
        - aushan_count_in_city: int - кол-во магазинов Ашан в городе
        - detmir_count_in_city: int - кол-во магазинов Детский мир в городе
        - lenta_count_in_city: int - кол-во магазинов Лента в городе

    Возвращает:
    predictions: numpy array с предсказанными значениями avg
    """

    # Загрузка артефактов
    loaded_artifacts = joblib.load('model_artifacts.joblib')

    loaded_pipeline = loaded_artifacts['prediction_pipeline']
    loaded_scaler = loaded_artifacts['scaler']
    loaded_model = loaded_artifacts['model']
    feature_columns = loaded_artifacts['feature_columns']

    print(f"\n📊 Входные данные: {len(input_data)} строк")
    print(f"📋 Входные колонки: {input_data.columns.tolist()}")

    # Применяем pipeline предобработки
    input_prepared = loaded_pipeline.transform(input_data)

    # One-hot encoding для chain
    input_encoded = pd.get_dummies(input_prepared, columns=['chain'], drop_first=True)

    # Выравниваем колонки
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Удаляем лишние колонки
    input_encoded = input_encoded[feature_columns]

    # Масштабирование
    input_scaled = loaded_scaler.transform(input_encoded)

    # Предсказание
    predictions = loaded_model.predict(input_scaled)

    return predictions


# ==================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ====================

print("\n" + "=" * 60)
print("ПРИМЕР ИСПОЛЬЗОВАНИЯ МОДЕЛИ")
print("=" * 60)

# Создаем тестовые данные с 8 признаками (БЕЗ city!)
test_input = pd.DataFrame([
    {
        'chain': 'Ашан',
        'cereals': 150.5,
        'milk': 200.3,
        'population': 8775735,
        'market_share': 0.25,
        'aushan_count_in_city': 15,
        'detmir_count_in_city': 8,
        'lenta_count_in_city': 12
    },
    {
        'chain': 'Лента',
        'cereals': 75.2,
        'milk': 100.1,
        'population': 335009,
        'market_share': 0.15,
        'aushan_count_in_city': 2,
        'detmir_count_in_city': 3,
        'lenta_count_in_city': 1
    },
    {
        'chain': 'Детский мир',
        'cereals': 95.0,
        'milk': 110.0,
        'population': 289503,
        'market_share': 0.10,
        'aushan_count_in_city': 1,
        'detmir_count_in_city': 2,
        'lenta_count_in_city': 0
    }
])

print("\n📝 Тестовые входные данные (8 признаков, БЕЗ city):")
print(test_input)

# Делаем предсказание
predictions = predict_from_input(test_input)

print("\n🎯 Предсказанные значения avg:")
for i, pred in enumerate(predictions):
    print(f"  {i + 1}. {pred:.2f}")

# Сохраняем пример входных данных
test_input.to_excel('example_input.xlsx', index=False)
print("\n✅ Пример входных данных сохранен в 'example_input.xlsx'")

print("\n" + "=" * 60)
print("📌 ТРЕБОВАНИЯ К ВХОДНЫМ ДАННЫМ ДЛЯ ПРЕДСКАЗАНИЯ:")
print("=" * 60)
print("JSON/Excel/DataFrame должен содержать 8 колонок:")
for i, col in enumerate(PREDICTION_INPUT_FEATURES, 1):
    print(f"  {i}. {col}")
