import torch
from tqdm.auto import tqdm

def train_one_epoch(model, dataloader, optimizer, device, epoch: int | None = None,):
    model.train()
    running = {"loss": 0.0, "ce": 0.0, "reg": 0.0, "acc": 0.0}
    num_batches = 0

    desc = f"Train [{epoch}]" if epoch is not None else "Train"
    pbar = tqdm(dataloader, desc=desc, leave=False)

    for points, labels in pbar:
        points = points.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss, logs = model.compute_loss(points, labels)
        loss.backward()
        optimizer.step()

        for k in running:
            running[k] += logs[k].item()
        num_batches += 1

    for k in running:
        running[k] /= num_batches

    return running  # dict with epoch averages

@torch.no_grad()
def evaluate(model, dataloader, device, epoch: int | None = None,):
    model.eval()
    running = {"loss": 0.0, "ce": 0.0, "reg": 0.0, "acc": 0.0}
    num_batches = 0

    desc = f"Val   [{epoch}]" if epoch is not None else "Val"
    pbar = tqdm(dataloader, desc=desc, leave=False)

    for points, labels in pbar:
        points = points.to(device)
        labels = labels.to(device)

        loss,logs = model.compute_loss(points, labels)

        for k in running:
            running[k] += logs[k].item()
        num_batches += 1

    for k in running:
        running[k] /= num_batches

    return running

def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=20,
    scheduler=None,
    checkpoint_path: str = "checkpoints/best_model.pt",
    # early stopping
    es_patience: int = 10,
    es_min_delta: float = 0.0,   # min improvement in val_loss to count
):
    
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    try:

        for epoch in range(1, num_epochs + 1):
            train_history= train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                epoch
            )

            val_history = evaluate(
                model,
                val_loader,
                device,
                epoch
            )

            # ----- early stopping logic (based on val_loss) -----
            # improvement if val_loss <= best_val_loss * (1 - es_min_delta)
            improved = val_history["loss"] <= best_val_loss * (1.0 - es_min_delta)

            if improved:
                best_val_loss = val_history["loss"]
                best_epoch = epoch
                epochs_no_improve = 0

                state = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                }
                torch.save(state, checkpoint_path)
                tag = " (saved)"
            else:
                epochs_no_improve += 1
                tag = f" (no improve {epochs_no_improve}/{es_patience})"

            if scheduler is not None:
                scheduler.step()

            print(
                f"Epoch {epoch:03d} | "
                f"train loss: {train_history['loss']:.4f}, acc: {train_history['acc']*100:.2f}% | "
                f"val loss: {val_history['loss']:.4f}, acc: {val_history['acc']*100:.2f}%{tag}"
            )

            # check patience
            if epochs_no_improve >= es_patience:
                print(
                    f"\n⏹ Early stopping: no val_loss improvement for "
                    f"{es_patience} consecutive epochs."
                )
                break

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user (Ctrl+C).")

    print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    return model