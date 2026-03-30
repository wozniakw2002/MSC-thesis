import torch
import os


def training_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    for img, gt_map in dataloader:
        img = img.to(device)
        gt_map = gt_map.to(device)

        pred_map = model(img)
        loss = criterion(pred_map, gt_map)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    return avg_loss


def evaluate_mae_mse(model, dataloader, device):
    model.eval()
    mae = 0

    with torch.no_grad():
        for img, gt_map in dataloader:
            img = img.to(device)
            gt_map = gt_map.to(device)

            pred_map = model(img)
            mae += abs(pred_map.sum() - gt_map.sum()).item()
            mse += ((pred_map.sum() - gt_map.sum())**2).item()
    mae = mae / len(dataloader)
    mse = mse / len(dataloader)
    return mae, mse


def train(
    epochs,
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    device,
    model_path,
    tolerance=float("inf")
):
    train_loss_list = []
    val_mae_list = []
    val_mse_list = []

    best_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_loss = training_epoch(
            model, train_dataloader, optimizer, criterion, device
        )

        val_mae, val_mse = evaluate_mae_mse(model, val_dataloader, device)

        train_loss_list.append(train_loss)
        val_mae_list.append(val_mae)
        val_mse_list.append(val_mse)

        print(
            f"epoch:{epoch} "
            f"loss:{train_loss:.4f} "
            f"mae:{val_mae:.4f} "
            f"best_mae:{best_mae:.4f}"
            f"mse:{val_mse:.4f} "
        )

        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(model.state_dict(), model_path)
            print("best model saved")
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement > tolerance:
                print(f"Early stopping triggered. ")
                break

        print()

    history = {
        "train_loss": train_loss_list,
        "val_mae": val_mae_list,
        "val_mse": val_mse_list,
        "best_mae": best_mae,
        "best_epoch": best_epoch,
    }

    return history