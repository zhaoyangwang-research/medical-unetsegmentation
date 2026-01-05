import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class NPZSegDataset(Dataset):
    def __init__(self, npz_path: str):
        z = np.load(npz_path)
        self.images = z["images"]
        self.masks = z["masks"]

        # Debug info
        print(f"Loaded {len(self.images)} samples from {npz_path}")
        print(f"Mask positive ratio: {self.masks.mean():.4f}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx].astype(np.float32) / 255.0  # (H,W,3)
        y = self.masks[idx].astype(np.float32)  # (H,W)

        x = torch.from_numpy(x).permute(2, 0, 1)  # (3,H,W)
        y = torch.from_numpy(y).unsqueeze(0)  # (1,H,W)
        return x, y


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x): return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.mid = DoubleConv(64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        m = self.mid(self.pool2(d2))
        u2 = self.up2(m)
        x = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(x)
        x = self.conv1(torch.cat([u1, d1], dim=1))
        return self.out(x)


def dice_from_logits(logits, y, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    inter = (preds * y).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()


def dice_loss(logits, y, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * y).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def combined_loss(logits, y, alpha=0.5):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y)
    dice = dice_loss(logits, y)
    return alpha * bce + (1 - alpha) * dice


@torch.no_grad()
def eval_dice(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        # dice_from_logits returns batch mean, so weight it
        bs = x.size(0)
        total += dice_from_logits(logits, y) * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def analyze_predictions(model, loader, device):
    """Debug function to check prediction statistics"""
    model.eval()
    all_preds = []
    all_targets = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        all_preds.append(preds.mean().item())
        all_targets.append(y.mean().item())

    avg_pred = np.mean(all_preds)
    avg_target = np.mean(all_targets)
    print(f"DEBUG: Avg prediction positive ratio: {avg_pred:.6f}")
    print(f"DEBUG: Avg target positive ratio: {avg_target:.6f}")
    return avg_pred, avg_target


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_npz", type=str,
                    default="dataprocessing/data/processed/kvasir_train.npz")

    ap.add_argument("--val_npz", type=str,
                    default="dataprocessing/data/processed/kvasir_validation.npz")

    ap.add_argument("--test_npz", type=str,
                    default="dataprocessing/data/processed/kvasir_test.npz")


    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loss", type=str, default="combined",
                    choices=["bce", "dice", "combined"])
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load datasets
    print("\n" + "=" * 50)
    print("Loading datasets...")
    train_ds = NPZSegDataset(args.train_npz)
    val_ds = NPZSegDataset(args.val_npz)
    test_ds = NPZSegDataset(args.test_npz)

    # Calculate class weight for BCE
    p = float(train_ds.masks.mean())
    pos_weight = torch.tensor([(1 - p) / max(p, 1e-6)], device=device)
    print(f"Positive weight for BCE: {pos_weight.item():.2f}")
    print("=" * 50)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Initialize model
    model = UNetSmall().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup loss function
    if args.loss == "bce":
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("Using BCE with pos_weight")
    elif args.loss == "dice":
        loss_fn = dice_loss
        print("Using Dice loss")
    else:  # combined
        bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        def combined_loss(logits, y, alpha=0.5):
            bce = bce_fn(logits, y)
            dice = dice_loss(logits, y)
            return alpha * bce + (1 - alpha) * dice

        loss_fn = combined_loss
        print("Using combined BCE + Dice loss")


    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=5
    )

    best_val = -1.0
    best_epoch = 0

    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)

    for epoch in range(1, args.epochs + 1):
        # Training phase
        model.train()
        train_losses = []
        train_dices = []

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            train_losses.append(loss.item())
            train_dices.append(dice_from_logits(logits, y))

            # Print batch progress every 10%
            if batch_idx % max(1, len(train_loader) // 10) == 0:
                print(f"Epoch {epoch:03d} | Batch {batch_idx:03d}/{len(train_loader):03d} | "
                      f"Loss: {loss.item():.4f} | Dice: {train_dices[-1]:.4f}")

        # Validation phase
        val_dice = eval_dice(model, val_loader, device)
        scheduler.step(val_dice)

        avg_train_loss = np.mean(train_losses)
        avg_train_dice = np.mean(train_dices)

        print(f"\nEpoch {epoch:03d}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Dice: {avg_train_dice:.4f} | "
              f"Val Dice: {val_dice:.4f}")
        print("-" * 50)

        # Analyze predictions (debug)
        if epoch == 1 or epoch % 5 == 0:
            analyze_predictions(model, val_loader, device)

        # Save best model
        if val_dice > best_val:
            best_val = val_dice
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_dice': val_dice,
            }, "dataprocessing/unet_small_best.pt")
            print(f"✓ Saved best model (Dice: {val_dice:.4f})")

    # Final evaluation
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)

    # Load best model
    checkpoint = torch.load("dataprocessing/unet_small_best.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final test
    test_dice = eval_dice(model, test_loader, device)

    print(f"\nRESULTS:")
    print(f"Best validation Dice: {best_val:.4f} (epoch {best_epoch})")
    print(f"Test Dice: {test_dice:.4f}")
    print(f"\nSaved best model: dataprocessing/unet_small_best.pt")

    # Final debug analysis
    print("\n" + "=" * 50)
    print("Final prediction analysis:")
    analyze_predictions(model, test_loader, device)


if __name__ == "__main__":
    main()