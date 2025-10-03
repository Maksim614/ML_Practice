from pathlib import Path
from typing import List, Dict

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

TARGET = "phone_type"

NUMERIC_COLS: List[str] = [
    "siblings_cnt",
    "taxi_trips_per_month",
    "years_to_change_phone",
    "budget_max",
    "charges_per_day",
]

ORDINAL_COLS: List[str] = [
    "residence_zone",
    "tech_affinity",
]

NOMINAL_COLS: List[str] = [
    "gender",
    "pc_os",
    "plays_mobile_games",
    "camera_is_important",
    "federal_district",
    "payment_method",
    "employment",
    "uses_smart_home",
    "works_in_it",
    "watch_type",
    "browser",
    "ui_customization_is_important",
    "materials_quality_is_important",
]

ALL_FEATURES = NUMERIC_COLS + ORDINAL_COLS + NOMINAL_COLS

RU_TO_SNAKE: Dict[str, str] = {
    "Какой у вас телефон? (Айфон - 1, Андроид - 2)": TARGET,
    "Пол (М - 1, Ж - 2)": "gender",
    "Количество братьев/сестёр (укажите цифру)": "siblings_cnt",
    "ОС на ПК (MacOS - 1, Windows - 2, Linux - 3)": "pc_os",
    "Среднее кол-во поездок на такси в месяц (укажите цифру)": "taxi_trips_per_month",
    "Играете в мобильные игры? (Да -1, Нет - 2)": "plays_mobile_games",
    "Область проживания ( в пределах садового - 1, ттк - 2, мцк - 3, мкад - 4, цкад - 5, московское большое кольцо - 6, дальше - 7": "residence_zone",
    "Важно ли качество камеры? (Да - 1, Нет - 2)": "camera_is_important",
    "Из какого ФО вы приехали ( цифры в порядке расположения списка на картинке)": "federal_district",
    "Чаще вы оплачиваете покупки... (картой - 1, стикером - 2, NFC - 3, наличными - 4, QR - 5)": "payment_method",
    "Как часто меняете телефон? (укажите среднее количество лет)": "years_to_change_phone",
    "Ваше положение (безработный - 1, частная компания - 2, госкомпания - 3)": "employment",
    "Пользуешься ли технологией умного дома? (Да - 1, Нет - 2)": "uses_smart_home",
    "Сфера работы IT? (Да - 1, Нет - 2)": "works_in_it",
    "Какие часы? (Нет часов - 1, Механические - 2, Электронные - 3)": "watch_type",
    "Какой максимальный бюджет готов потратить? (Введи число без пробелов)": "budget_max",
    "Сколько раз в день заряжаешь телефон? (Введи цифру)": "charges_per_day",
    "Каким браузером чаще пользуешься? (Google - 1, Яндекс - 2, Safari - 3, Opera - 4, Edge - 5, Firefox - 6)": "browser",
    "Любите ли вы новые технологии? ( по шкале от 1 до 5, где 1 - вообще не люблю, 3 - спокойно отношусь, а 5 - обожаю)": "tech_affinity",
    "Важна ли для вас возможность настройки интерфейса под себя? (Да - 1, Нет - 2)": "ui_customization_is_important",
    "Важно ли для вас качество материалов/материал корпуса? (Да -1, Нет - 2)": "materials_quality_is_important",
}

LABEL_NAME = {1: "iPhone", 2: "Android"}


# чтение CSV
def read_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    df = df.rename(columns=RU_TO_SNAKE)
    return df


# приведение типов
def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    cols = [TARGET] + ALL_FEATURES
    for col in cols:
        if col in df.columns:
            s = df[col].astype(str).str.replace("\xa0", " ", regex=False).str.replace(" ", "", regex=False)
            s = s.str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(s, errors="coerce")
    return df


def make_ohe() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


# pipeline: предобработка + KNN
def build_pipeline() -> Pipeline:
    numeric = NUMERIC_COLS + ORDINAL_COLS
    categorical = NOMINAL_COLS

    preproc = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", make_ohe())]), categorical),
        ]
    )

    return Pipeline([("prep", preproc), ("knn", KNeighborsClassifier())])


def get_neighbors_table(fitted_model: Pipeline, X: pd.DataFrame, y: pd.Series, df_orig: pd.DataFrame, idx: int,
                        k: int = 5):
    Z = fitted_model.named_steps["prep"].transform(X)
    knn = fitted_model.named_steps["knn"]
    n_query = min(len(X), k + 1)
    distances, indices = knn.kneighbors(Z[idx:idx + 1], n_neighbors=n_query)
    rows = []
    for j, d in zip(indices[0].tolist(), distances[0].tolist()):
        if j == idx:
            continue
        row = {"neighbor_idx": j, "distance": float(d), "true_label": int(y.iloc[j]),
               "label_name": LABEL_NAME.get(int(y.iloc[j]), str(y.iloc[j]))}
        for c in ["budget_max", "browser", "pc_os", "payment_method"]:
            if c in df_orig.columns:
                row[c] = df_orig.iloc[j][c]
        rows.append(row)
        if len(rows) >= k:
            break
    return pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)


def demo_neighbors(fitted_model: Pipeline, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, idx: int = 0, k: int = 5):
    pred = int(fitted_model.predict(X.iloc[[idx]])[0])
    proba = None
    if hasattr(fitted_model, "predict_proba"):
        p = fitted_model.predict_proba(X.iloc[[idx]])[0]
        classes = list(fitted_model.classes_)
        if 2 in classes:
            proba = float(p[classes.index(2)])
    print(f"\n=== Neighbors for row #{idx} ===")
    print(
        f"True: {LABEL_NAME[int(y.iloc[idx])]} | Pred: {LABEL_NAME[pred]} | P(Android)={proba if proba is not None else 'N/A'}")
    print(get_neighbors_table(fitted_model, X, y, df, idx=idx, k=k).to_string(index=False))


def main():
    csv_path = Path("data.csv")
    df = read_dataset(csv_path)
    df = coerce_types(df)

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].astype(int)

    min_class_count = int(y.value_counts().min())
    n_splits = min(5, min_class_count) if min_class_count >= 2 else 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    pipe = build_pipeline()
    grid = {
        "knn__n_neighbors": [1, 3, 5, 7, 9],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["minkowski"],
        "knn__p": [1, 2],
    }

    gs = GridSearchCV(pipe, param_grid=grid, scoring="balanced_accuracy", cv=cv, refit=True)
    gs.fit(X, y)

    best = gs.best_estimator_
    y_pred = cross_val_predict(best, X, y, cv=cv, method="predict")

    print("Best params:", gs.best_params_)
    print("CV balanced accuracy:", round(gs.best_score_, 3))
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y, y_pred, labels=[1, 2]))
    print("\nClassification report:")
    print(classification_report(y, y_pred, digits=3))

    best.fit(X, y)
    demo_neighbors(best, df, X, y, idx=0, k=5)


if __name__ == "__main__":
    main()
