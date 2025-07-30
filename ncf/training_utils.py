import torch
import torch.nn as nn
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, save_path='ncf_model.pt'):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_model(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_model(model)
            self.counter = 0

    def save_model(self, model):
        torch.save(model.state_dict(), self.save_path)
        print(f"Best model saved to {self.save_path}")

    def load_best_model(self, model):
        model.load_state_dict(torch.load(self.save_path))
        print(f"Best model loaded from {self.save_path}")


def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                early_stopper: EarlyStopping,
                device: torch.device,
                epochs: int) -> None:
    """
    Train and validate NCF model for N epochs, with checkpoints via early stopping
    """
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        for users, items, labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(users, items)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for users, items, labels in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                users = users.to(device)
                items = items.to(device)
                labels = labels.to(device)

                logits = model(users, items)
                total_val_loss += criterion(logits, labels).item()
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered")
            break
    # load the best model
    early_stopper.load_best_model(model)
