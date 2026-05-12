import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from preprocessing.build_cnn_dataset import build_cnn_dataset, load_subject_target_mapping
from models.deep.cnn_model import SimpleCNN1D, IMUDataset, train_one_epoch, evaluate


def main():
    # ---- load data ----
    mapping = load_subject_target_mapping("data/ankieta_score_and_target.csv")

    X, y, groups = build_cnn_dataset(
        raw_folder_path="data/cut_to_same_length",
        subject_target_mapping=mapping
    )

    print("Unikalne klasy:", np.unique(y))
    print("Rozkład klas:", {int(c): int((y == c).sum()) for c in np.unique(y)})

    # ---- group split ----
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    groups_train = groups[train_idx]
    groups_val = groups[val_idx]

    print("Train subjects:", np.unique(groups_train))
    print("Val subjects:", np.unique(groups_val))
    print("Train shape:", X_train.shape, y_train.shape)
    print("Val shape:", X_val.shape, y_val.shape)

    # ---- datasets ----
    train_dataset = IMUDataset(X_train, y_train)
    val_dataset = IMUDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # ---- model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(np.unique(y))

    model = SimpleCNN1D(n_classes=n_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # ---- training ----
    n_epochs = 20
    best_val_f1 = -1.0

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{n_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_cnn_model.pt")
            print("Zapisano najlepszy model.")

    print(f"Best val macro-F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
