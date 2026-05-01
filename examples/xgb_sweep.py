import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from batchsweep import experiment


def train(ctx, seed, max_depth, n_estimators):
    X_train, X_test, y_train, y_test = train_test_split(
        ctx.shared_data["X"],
        ctx.shared_data["y"],
        test_size=0.2,
        random_state=seed,
    )

    model = xgb.XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=seed,
        device=ctx.device,
        nthread=1,
    )
    model.fit(X_train, y_train)

    f1 = f1_score(y_test, model.predict(X_test), average="macro")
    return {"f1": f1}


if __name__ == "__main__":
    X, y = make_classification(
        n_samples=2000, n_features=20, n_classes=3, n_informative=10, random_state=0
    )

    experiment(
        fn=train,
        sweep={
            "seed": [1, 2, 3],
            "max_depth": [3, 5, 7],
            "n_estimators": [100, 300],
        },
        output_dir="./runs/xgb",
        workers_per_gpu=4,
        shared_data={
            "X": X.astype(np.float32),
            "y": y.astype(np.int64),
        },
    )
