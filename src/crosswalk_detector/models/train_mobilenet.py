"""Training helpers for MobileNetV3-Small on tile classification."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import random

from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from ..geo.pilot import repo_root


LABEL_TO_INDEX = {"no_crosswalk": 0, "crosswalk": 1}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}


@dataclass(frozen=True)
class DatasetRow:
    image_path: str
    label: str
    split: str
    tile_id: str


class TileDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(self, rows: list[DatasetRow], transform: transforms.Compose) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.rows[index]
        image = Image.open(row.image_path).convert("RGB")
        return self.transform(image), LABEL_TO_INDEX[row.label], row.tile_id


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_rows(labels_csv: Path) -> list[DatasetRow]:
    with labels_csv.open() as handle:
        reader = csv.DictReader(handle)
        rows = [
            DatasetRow(
                image_path=row["image_path"],
                label=row["label"],
                split=row["split"],
                tile_id=row.get("tile_id", row["relative_path"]),
            )
            for row in reader
        ]
    return rows


def _make_transforms(
    pretrained_normalization: bool = True,
    image_size: int = 128,
    stronger_augment: bool = False,
) -> tuple[transforms.Compose, transforms.Compose]:
    if pretrained_normalization:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        normalize_mean = weights.transforms().mean
        normalize_std = weights.transforms().std
    else:
        normalize_mean = [0.485, 0.456, 0.406]
        normalize_std = [0.229, 0.224, 0.225]
    train_steps: list[object]
    if stronger_augment:
        train_steps = [
            transforms.RandomResizedCrop(image_size, scale=(0.82, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.08, hue=0.02),
        ]
    else:
        train_steps = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        ]
    train_tf = transforms.Compose(
        [
            *train_steps,
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )
    return train_tf, eval_tf


def _loader(
    rows: list[DatasetRow],
    train: bool,
    batch_size: int = 32,
    pretrained_normalization: bool = True,
    image_size: int = 128,
    stronger_augment: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    train_tf, eval_tf = _make_transforms(pretrained_normalization, image_size, stronger_augment)
    dataset = TileDataset(rows, train_tf if train else eval_tf)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


class ScratchCrosswalkCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(256, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def _split_rows(run_name: str, export_name: str) -> tuple[list[DatasetRow], list[DatasetRow], list[DatasetRow]]:
    labels_csv = repo_root() / "data" / "processed" / run_name / "exports" / export_name / "labels.csv"
    rows = _load_rows(labels_csv)
    return (
        [row for row in rows if row.split == "train"],
        [row for row in rows if row.split == "val"],
        [row for row in rows if row.split == "test"],
    )


def train_mobilenet(
    run_name: str = "real-v1",
    export_name: str = "real-balanced-256",
    epochs: int = 6,
) -> dict[str, object]:
    labels_csv = repo_root() / "data" / "processed" / run_name / "exports" / export_name / "labels.csv"
    model_root = repo_root() / "models" / run_name / export_name
    model_root.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows, test_rows = _split_rows(run_name, export_name)

    train_loader = _loader(train_rows, train=True)
    val_loader = _loader(val_rows, train=False)
    test_loader = _loader(test_rows, train=False)

    device = _device()
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best_state = None
    best_val_acc = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * labels.size(0)
            predictions = logits.argmax(dim=1)
            train_correct += int((predictions == labels).sum().item())
            train_total += int(labels.size(0))

        val_acc = evaluate_model(model, val_loader, device)["accuracy"]
        train_acc = train_correct / max(1, train_total)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss / max(1, train_total), 6),
                "train_accuracy": round(train_acc, 6),
                "val_accuracy": round(val_acc, 6),
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(model, test_loader, device, include_predictions=True)
    model_path = model_root / "mobilenet_v3_small.pt"
    torch.save(model.state_dict(), model_path)

    metrics = {
        "run_name": run_name,
        "export_name": export_name,
        "model": "mobilenet_v3_small",
        "epochs": epochs,
        "device": device,
        "best_val_accuracy": round(best_val_acc, 6),
        "history": history,
        "test_accuracy": round(test_metrics["accuracy"], 6),
        "predictions": test_metrics["predictions"],
        "model_path": str(model_path),
    }
    (model_root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def train_scratch_cnn(
    run_name: str = "osm-v1-10k",
    export_name: str = "balanced-10k-v1",
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> dict[str, object]:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_root = repo_root() / "models" / run_name / export_name / "scratch-cnn"
    model_root.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows, test_rows = _split_rows(run_name, export_name)
    train_loader = _loader(train_rows, train=True, batch_size=batch_size, pretrained_normalization=False)
    val_loader = _loader(val_rows, train=False, batch_size=batch_size, pretrained_normalization=False)
    test_loader = _loader(test_rows, train=False, batch_size=batch_size, pretrained_normalization=False)

    device = _device()
    model = ScratchCrosswalkCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_state = None
    best_val_acc = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * labels.size(0)
            predictions = logits.argmax(dim=1)
            train_correct += int((predictions == labels).sum().item())
            train_total += int(labels.size(0))
        scheduler.step()

        val_acc = evaluate_model(model, val_loader, device)["accuracy"]
        train_acc = train_correct / max(1, train_total)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss / max(1, train_total), 6),
                "train_accuracy": round(train_acc, 6),
                "val_accuracy": round(float(val_acc), 6),
                "learning_rate": round(float(scheduler.get_last_lr()[0]), 8),
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            torch.save(best_state, model_root / "scratch_cnn_best.pt")
        print(
            f"epoch={epoch} train_acc={train_acc:.4f} val_acc={float(val_acc):.4f} "
            f"loss={train_loss / max(1, train_total):.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(model, test_loader, device, include_predictions=True)
    model_path = model_root / "scratch_cnn.pt"
    torch.save(model.state_dict(), model_path)
    metrics = {
        "run_name": run_name,
        "export_name": export_name,
        "model": "scratch_cnn",
        "pretrained": False,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": device,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "test_count": len(test_rows),
        "best_val_accuracy": round(best_val_acc, 6),
        "history": history,
        "test_accuracy": round(float(test_metrics["accuracy"]), 6),
        "predictions": test_metrics["predictions"],
        "model_path": str(model_path),
        "best_model_path": str(model_root / "scratch_cnn_best.pt"),
    }
    (model_root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def train_scratch_mobilenet(
    run_name: str = "osm-v2-50k",
    export_name: str = "balanced-50k-v1",
    epochs: int = 60,
    batch_size: int = 128,
    learning_rate: float = 8e-4,
    image_size: int = 160,
    seed: int = 42,
) -> dict[str, object]:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_root = repo_root() / "models" / run_name / export_name / "scratch-mobilenet-v3-small"
    model_root.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows, test_rows = _split_rows(run_name, export_name)
    worker_count = 4 if torch.cuda.is_available() else 0
    train_loader = _loader(
        train_rows,
        train=True,
        batch_size=batch_size,
        pretrained_normalization=False,
        image_size=image_size,
        stronger_augment=True,
        num_workers=worker_count,
    )
    val_loader = _loader(
        val_rows,
        train=False,
        batch_size=batch_size,
        pretrained_normalization=False,
        image_size=image_size,
        num_workers=worker_count,
    )
    test_loader = _loader(
        test_rows,
        train=False,
        batch_size=batch_size,
        pretrained_normalization=False,
        image_size=image_size,
        num_workers=worker_count,
    )

    device = _device()
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=8e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_state = None
    best_val_acc = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += float(loss.item()) * labels.size(0)
            predictions = logits.argmax(dim=1)
            train_correct += int((predictions == labels).sum().item())
            train_total += int(labels.size(0))
        scheduler.step()

        val_acc = evaluate_model(model, val_loader, device)["accuracy"]
        train_acc = train_correct / max(1, train_total)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss / max(1, train_total), 6),
                "train_accuracy": round(train_acc, 6),
                "val_accuracy": round(float(val_acc), 6),
                "learning_rate": round(float(scheduler.get_last_lr()[0]), 8),
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            torch.save(best_state, model_root / "scratch_mobilenet_best.pt")
        print(
            f"epoch={epoch} train_acc={train_acc:.4f} val_acc={float(val_acc):.4f} "
            f"loss={train_loss / max(1, train_total):.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(model, test_loader, device, include_predictions=True)
    model_path = model_root / "scratch_mobilenet.pt"
    torch.save(model.state_dict(), model_path)
    metrics = {
        "run_name": run_name,
        "export_name": export_name,
        "model": "mobilenet_v3_small",
        "pretrained": False,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "image_size": image_size,
        "seed": seed,
        "device": device,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "test_count": len(test_rows),
        "best_val_accuracy": round(best_val_acc, 6),
        "history": history,
        "test_accuracy": round(float(test_metrics["accuracy"]), 6),
        "predictions": test_metrics["predictions"],
        "model_path": str(model_path),
        "best_model_path": str(model_root / "scratch_mobilenet_best.pt"),
    }
    (model_root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    include_predictions: bool = False,
) -> dict[str, object]:
    model.eval()
    correct = 0
    total = 0
    predictions_output: list[dict[str, object]] = []
    with torch.no_grad():
        for images, labels, tile_ids in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.size(0))
            if include_predictions:
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
                for tile_id, pred, label, prob in zip(
                    tile_ids,
                    predictions.cpu().tolist(),
                    labels.cpu().tolist(),
                    probs,
                ):
                    predictions_output.append(
                        {
                            "tile_id": tile_id,
                            "predicted_label": INDEX_TO_LABEL[pred],
                            "true_label": INDEX_TO_LABEL[label],
                            "crosswalk_probability": round(float(prob), 6),
                        }
                    )
    payload: dict[str, object] = {"accuracy": correct / max(1, total)}
    if include_predictions:
        payload["predictions"] = predictions_output
    return payload
