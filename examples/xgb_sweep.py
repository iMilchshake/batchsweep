import torch
import xgboost as xgb
from sklearn.metrics import f1_score

from batchsweep import experiment


def train(ctx, seed, max_depth, n_estimators):
    X_train = torch.load("data/X_train.pt").numpy()
    y_train = torch.load("data/y_train.pt").numpy()
    X_test = torch.load("data/X_test.pt").numpy()
    y_test = torch.load("data/y_test.pt").numpy()

    model = xgb.XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=seed,
        device=ctx.device,
    )
    model.fit(X_train, y_train)

    f1 = f1_score(y_test, model.predict(X_test), average="macro")
    ctx.log(f"f1={f1:.4f}")
    return {"f1": f1}


if __name__ == "__main__":
    experiment(
        fn=train,
        sweep={
            "seed": [1, 2, 3],
            "max_depth": [3, 5, 7],
            "n_estimators": [100, 300],
        },
        output_dir="./runs/xgb",
        workers_per_gpu=2,
    )
