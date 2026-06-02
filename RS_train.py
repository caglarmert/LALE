import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from segmentation_models_pytorch.losses import DiceLoss
from albumentations.pytorch import ToTensorV2
import wandb
from typing import Optional, Callable, Tuple
from datasets import load_dataset 

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        if x.dim() == 4:
            return (x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + self.eps)) * self.weight.view(1, -1, 1, 1)
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * self.weight

class StarReLU(nn.Module):
    def __init__(self, scale_value=0.8944, bias_value=-0.4472, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(torch.tensor(scale_value))
        self.bias = nn.Parameter(torch.tensor(bias_value))
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias

class ConvMixerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = RMSNorm(dim)
        self.pwconv1 = nn.Conv2d(dim, dim * 4, kernel_size=1)
        self.act = StarReLU()
        self.pwconv2 = nn.Conv2d(dim * 4, dim, kernel_size=1)
    def forward(self, x):
        inputs = x
        x = self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(x)))))
        return x + inputs

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp_fc1 = nn.Linear(dim, hidden_dim)
        self.mlp_dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.mlp_act = StarReLU()
        self.mlp_drop1, self.mlp_drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.mlp_fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, H, W):
        B, C, _, _ = x.shape
        x_seq = x.flatten(2).transpose(1, 2)
        normed_x = self.norm1(x_seq)
        attn_out, _ = self.attn(normed_x, normed_x, normed_x)
        x_seq = x_seq + attn_out

        mx = self.norm2(x_seq)
        mx = self.mlp_fc1(mx).transpose(1, 2).view(B, -1, H, W)
        mx = self.mlp_drop2(self.mlp_fc2(self.mlp_drop1(self.mlp_act(self.mlp_dwconv(mx).flatten(2).transpose(1, 2)))))
        return (x_seq + mx).transpose(1, 2).view(B, C, H, W)

class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.norm = RMSNorm(out_dim)
    def forward(self, x): return self.norm(self.proj(x))

class MLPDecoder(nn.Module):
    def __init__(self, encoder_dims, decoder_dim, num_classes):
        super().__init__()
        self.linear_c = nn.ModuleList([nn.Conv2d(dim, decoder_dim, kernel_size=1) for dim in encoder_dims])
        self.linear_fuse = nn.Sequential(nn.Conv2d(decoder_dim * len(encoder_dims), decoder_dim, kernel_size=1, bias=False), nn.BatchNorm2d(decoder_dim), StarReLU())
        self.dropout = nn.Dropout2d(0.1)
        self.classifier = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, features):
        target_size = features[0].shape[2:] 
        outs = [F.interpolate(self.linear_c[i](feat), size=target_size, mode='bilinear', align_corners=False) if i > 0 else self.linear_c[i](feat) for i, feat in enumerate(features)]
        return self.classifier(self.dropout(self.linear_fuse(torch.cat(outs, dim=1))))

class NanoRSFormer(nn.Module):
    def __init__(self, in_chans=3, num_classes=1, dims=[32, 64, 128, 256], depths=[1,1,1,1], num_heads=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_chans, dims[0] // 2, kernel_size=3, stride=2, padding=1), RMSNorm(dims[0] // 2), StarReLU(), nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1), RMSNorm(dims[0]))
        self.stages, self.downsamples = nn.ModuleList(), nn.ModuleList()
        for i in range(4):
            self.downsamples.append(Downsample(dims[i-1], dims[i]) if i > 0 else nn.Identity())
            self.stages.append(nn.ModuleList([ConvMixerBlock(dims[i]) if i < 2 else TransformerBlock(dims[i], num_heads=num_heads) for _ in range(depths[i])]))
        self.decoder = MLPDecoder(encoder_dims=dims, decoder_dim=128, num_classes=num_classes)

    def forward(self, x):
        orig_H, orig_W = x.shape[2:]
        features, x = [], self.stem(x)
        for i in range(4):
            x = self.downsamples[i](x)
            for block in self.stages[i]: x = block(x, x.shape[2], x.shape[3]) if isinstance(block, TransformerBlock) else block(x)
            features.append(x)
        return F.interpolate(self.decoder(features), size=(orig_H, orig_W), mode='bilinear', align_corners=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=str, default="LALE")
    parser.add_argument("--no-save", action="store_true", help="Disable model saving and loading")
    return parser.parse_args()

args = parse_args()

# Configuration
config_dict = {
    # Data
    "DATASET_NAME": "caglarmert/ARAS400k-cnet",
    "NUM_CLASSES": 7,

    # Training
    "EPOCHS": 10,
    "BATCH_SIZE": 128,
    "PATIENCE": 5,
    "LEARNING_RATE": 1e-4,
    "GRAD_CLIP": 1.0,

    # Model
    "ARCHITECTURE": args.architecture,

    # DataLoader
    "NUM_WORKERS": 16,
    "PIN_MEMORY": True,
}

print(config_dict)

COLOR2LABEL = {
    (0, 100, 0): 0,      # Tree
    (255, 182, 193): 1,  # Shrub
    (154, 205, 50): 2,   # Grass
    (255, 215, 0): 3,    # Crop
    (139, 69, 19): 4,    # Built-up
    (211, 211, 211): 5,  # Barren
    (0, 0, 255): 6,      # Water
}

LABEL2COLOR = {k: list(v) for k, v in enumerate(COLOR2LABEL.keys())}
WANDB_CLASSES = {0:"Tree",1:"Shrub",2:"Grass",3:"Crop",4:"Built-up",5:"Barren",6:"Water"}

# Fast LUT for Mask Conversion
COLOR_LUT = {
    (0 << 16) | (100 << 8) | 0: 0,
    (255 << 16) | (182 << 8) | 193: 1,
    (154 << 16) | (205 << 8) | 50: 2,
    (255 << 16) | (215 << 8) | 0: 3,
    (139 << 16) | (69 << 8) | 19: 4,
    (211 << 16) | (211 << 8) | 211: 5,
    (0 << 16) | (0 << 8) | 255: 6,
}

train_transforms = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


def mask_to_class(mask: np.ndarray) -> np.ndarray:
    """Converts an RGB mask into a single-channel class mask efficiently using bitwise packing."""
    mask_32 = mask.astype(np.uint32)
    packed = (mask_32[..., 0] << 16) | (mask_32[..., 1] << 8) | mask_32[..., 2]

    out = np.zeros(mask.shape[:2], dtype=np.uint8)

    for k, v in COLOR_LUT.items():
        out[packed == k] = v

    return out


class HFSegmentationDataset(Dataset):
    """Memory-efficient standard dataset utilizing local Arrow cache."""
    def __init__(self, hf_dataset, transform: Optional[Callable] = None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Convert PIL Images to NumPy RGB structures
        image = np.array(item["image"].convert("RGB"))
        mask_rgb = np.array(item["conditioning_image"].convert("RGB"))
        
        mask = mask_to_class(mask_rgb)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
            mask = mask.long()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask


class ConfusionMatrixTracker:
    """Computes segmentation metrics on a rolling basis with zero RAM overhead."""
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, labels: np.ndarray, preds: np.ndarray):
        mask = (labels >= 0) & (labels < self.num_classes)
        self.mat += np.bincount(
            self.num_classes * labels[mask].astype(int) + preds[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def compute_metrics(self):
        diag = np.diag(self.mat)
        sum_over_rows = self.mat.sum(axis=1) # Ground Truths per class
        sum_over_cols = self.mat.sum(axis=0) # Predictions per class

        precision = np.where(sum_over_cols > 0, diag / sum_over_cols, 0.0)
        recall = np.where(sum_over_rows > 0, diag / sum_over_rows, 0.0)
        f1 = np.where((precision + recall) > 0, 2 * (precision * recall) / (precision + recall), 0.0)
        
        denominator = sum_over_rows + sum_over_cols - diag
        iou = np.where(denominator > 0, diag / denominator, 0.0)

        accuracy = diag.sum() / self.mat.sum() if self.mat.sum() > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision_macro": np.mean(precision),
            "recall_macro": np.mean(recall),
            "f1_macro": np.mean(f1),
            "iou_macro": np.mean(iou),
            "per_class_iou": iou,
            "per_class_f1": f1
        }


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (imgs, masks) in enumerate(loader):
            imgs, masks = imgs.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(self.device.type == 'cuda')):
                outputs = self.model(imgs)
                loss = self.criterion(outputs, masks)

            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config_dict["GRAD_CLIP"])
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 1000 == 0:
                print(f"  Batch {batch_idx + 1}, Current Loss: {loss.item():.4f}")

        return total_loss / max(1, num_batches)


def evaluate(model, loader, num_classes, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    criterion = DiceLoss(mode="multiclass")
    tracker = ConfusionMatrixTracker(num_classes)

    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(loader):
            imgs, masks = imgs.to(device), masks.to(device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                outputs = model(imgs)
                loss = criterion(outputs, masks)

            total_loss += loss.item()
            num_batches += 1

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            ground_truth = masks.cpu().numpy()

            tracker.update(ground_truth, preds)

            if (batch_idx + 1) % 1000 == 0:
                print(f"  Eval Batch {batch_idx + 1}")

    computed = tracker.compute_metrics()
    
    metrics = {
        "accuracy": computed["accuracy"],
        "iou": computed["iou_macro"],
        "precision": computed["precision_macro"],
        "recall": computed["recall_macro"],
        "f1": computed["f1_macro"],
        "loss": total_loss / max(1, num_batches)
    }

    for i, score in enumerate(computed["per_class_iou"]):
        metrics[f"iou_class_{i}"] = score
    for i, score in enumerate(computed["per_class_f1"]):
        metrics[f"f1_class_{i}"] = score

    return metrics


def upload_sanity_checks(model, raw_hf_dataset, device, num_samples=5):
    """Grabs a few elements safely from the dataset for WandB overlays."""
    model.eval()
    print(f"Uploading {num_samples} evaluation sanity checks to WandB...")
    
    wandb_images = []
    
    # Take a small distinct slice from the raw map dataset
    sampled_items = [raw_hf_dataset[i] for i in range(num_samples)]
    
    with torch.no_grad():
        for idx, hf_item in enumerate(sampled_items):
            original_img = np.array(hf_item["image"].convert("RGB"))
            mask_rgb = np.array(hf_item["conditioning_image"].convert("RGB"))
            ground_truth = mask_to_class(mask_rgb)
            
            augmented = val_transforms(image=original_img, mask=ground_truth)
            image_tensor = augmented["image"].unsqueeze(0).to(device)
            
            # Tier 1 Fix: Enforce bfloat16 in sanity check outputs
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                output = model(image_tensor)
                
            pred = torch.argmax(F.softmax(output, dim=1), dim=1).squeeze(0).cpu().numpy()
            
            wandb_img = wandb.Image(
                original_img,
                masks={
                    "predictions": {
                        "mask_data": pred,
                        "class_labels": WANDB_CLASSES
                    },
                    "ground_truth": {
                        "mask_data": ground_truth,
                        "class_labels": WANDB_CLASSES
                    }
                },
                caption=f"Test Sample {idx}"
            )
            wandb_images.append(wandb_img)
            
    wandb.log({"sanity_checks": wandb_images})


def main():
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(
        project="ARAS1M_Segmentation",
        entity="caglarmert",
        config=config_dict
    )

    print(f"Loading Hugging Face dataset splits from {config_dict['DATASET_NAME']}...")
    hf_train = load_dataset(config_dict["DATASET_NAME"], split="train")
    hf_val = load_dataset(config_dict["DATASET_NAME"], split="validation")
    hf_test = load_dataset(config_dict["DATASET_NAME"], split="test")

    train_ds = HFSegmentationDataset(hf_train, train_transforms)
    val_ds = HFSegmentationDataset(hf_val, val_transforms)
    test_ds = HFSegmentationDataset(hf_test, val_transforms)

    # Dataloaders with optimized worker logic
    train_loader = DataLoader(
        train_ds, batch_size=config_dict["BATCH_SIZE"], shuffle=True,
        num_workers=config_dict["NUM_WORKERS"], pin_memory=config_dict["PIN_MEMORY"],
        persistent_workers=True, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=config_dict["BATCH_SIZE"], shuffle=False,
        num_workers=config_dict["NUM_WORKERS"], pin_memory=config_dict["PIN_MEMORY"],
        persistent_workers=True, prefetch_factor=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=config_dict["BATCH_SIZE"], shuffle=False,
        num_workers=config_dict["NUM_WORKERS"], pin_memory=config_dict["PIN_MEMORY"],
        persistent_workers=True, prefetch_factor=4
    )

    # Initializing custom architecture instead of smp library
    model = NanoRSFormer(
        in_chans=3, 
        num_classes=config_dict["NUM_CLASSES"]
    ).to(device)

    model = torch.compile(model)

    print(f"Model: {config_dict['ARCHITECTURE']}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = DiceLoss(mode="multiclass")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config_dict["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.1
    )

    trainer = Trainer(model, criterion, optimizer, scheduler, device)

    best_f1 = 0.0
    counter = 0
    model_save_path = f"best_{config_dict['ARCHITECTURE']}.pth"

    print("\nStarting training...")

    for epoch in range(config_dict["EPOCHS"]):
        print(f"\nEpoch {epoch + 1}/{config_dict['EPOCHS']}")

        print("Training...")
        train_loss = trainer.train_epoch(train_loader)

        print("Validating...")
        val_metrics = evaluate(model, val_loader, config_dict["NUM_CLASSES"], device)
        current_lr = optimizer.param_groups[0]['lr']

        scheduler.step(val_metrics["f1"])

        log_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "learning_rate": current_lr,
            **{f"val_{k}": v for k, v in val_metrics.items()}
        }
        wandb.log(log_data)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}, Val IoU: {val_metrics['iou']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            counter = 0
            if not args.no_save:
                torch.save(model.state_dict(), model_save_path)
                print(f"New best model saved to {model_save_path} with F1: {best_f1:.4f}")
            else:
                print(f"New best F1: {best_f1:.4f} (saving disabled)")
        else:
            counter += 1
            print(f"No improvement for {counter}/{config_dict['PATIENCE']} epochs")
            if counter >= config_dict["PATIENCE"]:
                print(f"Early stopping after {epoch + 1} epochs")
                break

    print("\n" + "=" * 50)
    print("Testing Best Model")
    print("=" * 50)

    if not args.no_save:
        model.load_state_dict(torch.load(model_save_path, weights_only=True))

    print("Running test evaluation...")
    test_metrics = evaluate(model, test_loader, config_dict["NUM_CLASSES"], device)

    print("\nTest Results:")
    for k, v in test_metrics.items():
        if k.startswith('iou_class') or k.startswith('f1_class'):
            continue
        print(f"  {k}: {v:.4f}")

    wandb.log({f"test_{k}": v for k, v in test_metrics.items()})

    upload_sanity_checks(model, hf_test, device, num_samples=5)

    wandb.finish()
    print("Training process completed fully!")


if __name__ == "__main__":
    main()