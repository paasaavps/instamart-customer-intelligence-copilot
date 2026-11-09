
"""Part B: RFM segmentation and a leakage-aware lapse model."""

from pathlib import Path
from sklearn.base import clone
import numpy as np
import pandas as pd
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data/clean"
OUT = ROOT / "outputs"

SNAPSHOT = pd.Timestamp("2025-10-01")


def load():
    c = pd.read_csv(CLEAN / "customers.csv")
    o = pd.read_csv(CLEAN / "orders.csv")
    o["order_ts"] = pd.to_datetime(o.order_ts)
    return c, o


def rfm_and_segments(k=4):
    c, o = load()
    done = o[o.status.eq("completed")].copy()

    rfm = done.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()

    rfm["recency"] = (SNAPSHOT - rfm.last_order).dt.days
    rfm = rfm.drop(columns="last_order")

    scaler = StandardScaler()
    X = scaler.fit_transform(
        rfm[["recency", "frequency", "monetary"]]
    )

    diagnostics = []

    from sklearn.metrics import silhouette_score

    for n in range(2, 9):
        km = KMeans(
            n_clusters=n,
            n_init=10,
            random_state=42
        ).fit(X)

        diagnostics.append({
            "k": n,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, km.labels_)
        })

    km = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    rfm["segment_id"] = km.fit_predict(X)

    profile = rfm.groupby("segment_id").agg(
        size=("customer_id", "size"),
        recency=("recency", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean")
    ).reset_index()

    order = profile.sort_values(
        "monetary",
        ascending=False
    ).segment_id.tolist()

    # Give every cluster a distinct business-friendly name.
    # Highest monetary segment is Premium.
    # Highest frequency among remaining segments is Frequent Loyal.
    # Lowest monetary among remaining segments is At Risk.
    # The remaining segment is Regular Low-Value.

    remaining = [i for i in profile.segment_id if i != order[0]]

    lowest_value = profile.loc[
        profile.segment_id.isin(remaining)
    ].sort_values("monetary").iloc[0]["segment_id"]

    remaining_after_lowest = [
        i for i in remaining if i != lowest_value
    ]

    highest_frequency = profile.loc[
        profile.segment_id.isin(remaining_after_lowest)
    ].sort_values("frequency", ascending=False).iloc[0]["segment_id"]

    name_map = {order[0]: "Premium"}
    name_map[highest_frequency] = "Frequent Loyal"
    name_map[lowest_value] = "At Risk"

    for i in profile.segment_id:
        if i not in name_map:
            name_map[i] = "Regular Low-Value"

    rfm["segment_name"] = rfm.segment_id.map(name_map)
    rfm["is_premium"] = rfm.segment_id.eq(order[0])

    OUT.mkdir(exist_ok=True)

    rfm[
        [
            "customer_id",
            "recency",
            "frequency",
            "monetary",
            "segment_id",
            "segment_name",
            "is_premium"
        ]
    ].to_csv(OUT / "customer_segments.csv", index=False)

    pca = PCA(n_components=2).fit(X)

    return (
        rfm,
        pd.DataFrame(diagnostics),
        profile,
        pca.explained_variance_ratio_
    )


def build_lapse_model():

    c, o = load()

    cutoff = pd.Timestamp("2025-09-01")
    end = pd.Timestamp("2025-10-01")

    # Build historical features using orders before September.
    hist = o[o.order_ts < cutoff].copy()
    done = hist[hist.status.eq("completed")]

    cohort = done.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount=("discount_pct", "mean"),
        avg_delivery=("delivery_minutes", "mean")
    ).reset_index()

    cohort["recency"] = (cutoff - cohort.last_order).dt.days
    cohort = cohort.drop(columns="last_order")

    all_final = (
        hist[hist.status.isin(["completed", "cancelled"])]
        .groupby("customer_id")
        .status
        .agg(
            total="count",
            cancelled=lambda x: (x == "cancelled").sum()
        )
        .reset_index()
    )

    cohort = cohort.merge(
        all_final,
        on="customer_id",
        how="left"
    )

    cohort["cancellation_rate"] = (
        cohort.cancelled / cohort.total
    )

    # Favourite product category.
    cats = (
        done.merge(
            pd.read_csv(CLEAN / "products.csv")[
                ["product_id", "category"]
            ],
            on="product_id"
        )
        .groupby(["customer_id", "category"])
        .size()
        .rename("n")
        .reset_index()
    )

    fav = (
        cats.sort_values(
            ["customer_id", "n", "category"],
            ascending=[True, False, True]
        )
        .drop_duplicates("customer_id")
        .rename(columns={"category": "favourite_category"})[
            ["customer_id", "favourite_category"]
        ]
    )

    cohort = (
        cohort.merge(fav, on="customer_id", how="left")
        .merge(c, on="customer_id", how="left")
    )

    cohort["signup_date"] = pd.to_datetime(cohort.signup_date)

    cohort["tenure_days"] = (
        cutoff - cohort.signup_date
    ).dt.days

    # Label customers based on completed orders during September.
    post = (
        o[
            (o.order_ts >= cutoff)
            & (o.order_ts < end)
            & o.status.eq("completed")
        ]
        .groupby("customer_id")
        .size()
    )

    cohort["lapsed"] = (
        ~cohort.customer_id.isin(post.index)
    ).astype(int)

    cohort = cohort.drop(
        columns=["signup_date", "total", "cancelled"]
    )

    features = [
        "recency",
        "frequency",
        "monetary",
        "avg_discount",
        "cancellation_rate",
        "avg_delivery",
        "favourite_category",
        "city",
        "age",
        "gender",
        "tenure_days"
    ]

    X = cohort[features].copy()
    y = cohort.lapsed

    numeric = [
        "recency",
        "frequency",
        "monetary",
        "avg_discount",
        "cancellation_rate",
        "avg_delivery",
        "age",
        "tenure_days"
    ]

    categorical = [
        "favourite_category",
        "city",
        "gender"
    ]

    # Display missing values in the raw features.
    # They will be imputed inside the Pipeline.
    print("\nMissing values in raw features:")
    print(X.isna().sum().to_string())

    # Train/test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Oversample only the minority class in training data.
    class_counts = y_train.value_counts()

    minority_class = class_counts.idxmin()
    majority_count = int(class_counts.max())
    minority_count = int(class_counts.min())

    X_minority = X_train[y_train == minority_class]
    y_minority = y_train[y_train == minority_class]

    n_extra = majority_count - minority_count

    X_extra = X_minority.sample(
        n=n_extra,
        replace=True,
        random_state=42
    )

    y_extra = y_minority.loc[X_extra.index]

    X_train_balanced = pd.concat([X_train, X_extra])
    y_train_balanced = pd.concat([y_train, y_extra])

    print("\nBalanced training counts:")
    print(y_train_balanced.value_counts())

    print("\nOriginal training class counts:")
    print(y_train.value_counts())

    # Preprocessing.
    # Imputation, scaling and encoding are fitted only on training data.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="unknown"
        )),
        ("encoder", OneHotEncoder(
            drop="first",
            handle_unknown="ignore"
        ))
    ])

    prep = ColumnTransformer([
        ("num", numeric_pipeline, numeric),
        ("cat", categorical_pipeline, categorical)
    ])

    # Train and tune both models using original training data.
    models = {}

    for name, est, grid in [
        (
            "logistic",
            LogisticRegression(max_iter=1000),
            {"model__C": [0.01, 0.1, 1, 10, 100]}
        ),
        (
            "forest",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),
            {
                "model__max_depth": [3, 5, 8],
                "model__min_samples_leaf": [5, 10, 20]
            }
        )
    ]:

        pipe = Pipeline([
            ("prep", prep),
            ("model", est)
        ])

        search = GridSearchCV(
            pipe,
            grid,
            cv=5,
            scoring="f1",
            n_jobs=-1
        )

        search.fit(X_train, y_train)

        pred = search.predict(X_test)

        models[name] = {
            "best_params": search.best_params_,
            "cv_f1": search.best_score_,
            "precision": precision_score(
                y_test, pred, zero_division=0
            ),
            "recall": recall_score(
                y_test, pred, zero_division=0
            ),
            "f1": f1_score(
                y_test, pred, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_test, pred
            ).tolist(),
            "search": search
        }

    print("\nUnbalanced model comparison (positive class = lapsed):")

    for name, result in models.items():
        print(f"\n{name.title()} best parameters:", result["best_params"])
        print(f"CV F1: {result['cv_f1']:.4f}")
        print(f"Test precision: {result['precision']:.4f}")
        print(f"Test recall: {result['recall']:.4f}")
        print(f"Test F1: {result['f1']:.4f}")
        print("Test confusion matrix:")
        print(np.array(result["confusion_matrix"]))

    # Random Forest feature importance.
    forest_pipeline = models["forest"]["search"].best_estimator_

    forest_features = (
        forest_pipeline.named_steps["prep"].get_feature_names_out()
    )

    forest_importances = (
        forest_pipeline.named_steps["model"].feature_importances_
    )

    rf_importance = pd.DataFrame({
        "feature": forest_features,
        "importance": forest_importances
    }).sort_values("importance", ascending=False)

    print("\nTop 5 Random Forest Features:")
    print(rf_importance.head(5).to_string(index=False))

    # Original Logistic Regression coefficients.
    logistic_pipeline = models["logistic"]["search"].best_estimator_

    # Verify transformed training and test features have no missing values.
    transformed_train = logistic_pipeline.named_steps["prep"].transform(X_train)
    transformed_test = logistic_pipeline.named_steps["prep"].transform(X_test)

    def has_missing_values(values):
        if hasattr(values, "tocsr"):
            return bool(np.isnan(values.data).any())
        return bool(np.isnan(np.asarray(values)).any())

    assert not has_missing_values(transformed_train)
    assert not has_missing_values(transformed_test)

    print("\nAll transformed feature columns are free of missing values.")

    # Compare training-only scaler mean with full-cohort frequency mean.
    numeric_pipeline_fitted = (
        logistic_pipeline.named_steps["prep"]
        .named_transformers_["num"]
    )

    frequency_index = numeric.index("frequency")

    training_scaler_mean = (
        numeric_pipeline_fitted.named_steps["scaler"]
        .mean_[frequency_index]
    )

    full_cohort_mean = X["frequency"].mean()

    print("\nFrequency Mean Comparison:")
    print("Training scaler mean (training rows only):", training_scaler_mean)
    print("Full-cohort frequency mean:", full_cohort_mean)

    logistic_features = (
        logistic_pipeline.named_steps["prep"].get_feature_names_out()
    )

    logistic_coefficients = (
        logistic_pipeline.named_steps["model"].coef_[0]
    )

    lr_coefficients = pd.DataFrame({
        "feature": logistic_features,
        "coefficient": logistic_coefficients
    })

    lr_coefficients["absolute_value"] = (
        lr_coefficients.coefficient.abs()
    )

    print("\nTop 3 Logistic Regression Coefficients:")

    print(
        lr_coefficients.sort_values(
            "absolute_value",
            ascending=False
        ).head(3)[
            ["feature", "coefficient"]
        ].to_string(index=False)
    )

    balanced_logistic = clone(
        models["logistic"]["search"].best_estimator_
    )

    balanced_logistic.fit(
        X_train_balanced,
        y_train_balanced
    )

    balanced_pred = balanced_logistic.predict(X_test)

    balanced_metrics = {
        "precision": precision_score(
            y_test, balanced_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, balanced_pred, zero_division=0
        ),
        "f1": f1_score(
            y_test, balanced_pred, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, balanced_pred
        ).tolist()
    }

    print("\nBalanced Logistic Regression:")
    print("Precision:", balanced_metrics["precision"])
    print("Recall:", balanced_metrics["recall"])
    print("F1:", balanced_metrics["f1"])
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, balanced_pred))

    # Select Balanced Logistic Regression as final model.
    selected = balanced_logistic

    # Save the selected model.
    OUT.mkdir(exist_ok=True)

    joblib.dump(
        selected,
        OUT / "lapse_model.joblib"
    )

    # Reload and verify identical predictions.
    reloaded_model = joblib.load(
        OUT / "lapse_model.joblib"
    )

    assert np.array_equal(
        selected.predict(X_test),
        reloaded_model.predict(X_test)
    )

    print("\nModel saved and reload verification passed.")

    # Generate Premium customer retention list.
    # Premium flag is post-hoc and is used only for this illustrative list.
    segments, _, _, _ = rfm_and_segments()

    premium = set(
        segments.loc[
            segments.is_premium,
            "customer_id"
        ]
    )

    probs = selected.predict_proba(X_test)[
        :, list(selected.classes_).index(1)
    ]

    ret = pd.DataFrame({
        "customer_id": cohort.loc[
            X_test.index, "customer_id"
        ].values,
        "lapse_probability": probs,
        "actually_lapsed": y_test.values
    })

    ret = (
        ret[ret.customer_id.isin(premium)]
        .sort_values(
            "lapse_probability",
            ascending=False
        )
        .head(10)
    )

    ret.to_csv(
        OUT / "retention_list.csv",
        index=False
    )

    return {
        "cohort_rows": len(cohort),
        "lapsed": int(y.sum()),
        "baseline_accuracy": accuracy_score(
            y_test,
            np.zeros(len(y_test), dtype=int)
        ),
        "models": {
            k: {
                a: b
                for a, b in v.items()
                if a != "search"
            }
            for k, v in models.items()
        },
        "balanced_logistic": balanced_metrics,
        "retention_list": ret
    }
