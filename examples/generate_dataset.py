import torch
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    X, y = make_classification(
        n_samples=2000, n_features=20, n_classes=3, n_informative=10, random_state=0
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    Path("data").mkdir(exist_ok=True)
    torch.save(torch.tensor(X_train, dtype=torch.float32), "data/X_train.pt")
    torch.save(torch.tensor(y_train, dtype=torch.long), "data/y_train.pt")
    torch.save(torch.tensor(X_test, dtype=torch.float32), "data/X_test.pt")
    torch.save(torch.tensor(y_test, dtype=torch.long), "data/y_test.pt")

    print(
        f"Saved dataset: {len(X_train)} train / {len(X_test)} test samples, {X.shape[1]} features, 3 classes."
    )
