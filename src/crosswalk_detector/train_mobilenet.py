"""Training helpers for MobileNetV3-Small on tile classification."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .pilot import repo_root


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
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
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


def _make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    weights = MobileNet_V3_Small_Weights.DEFAULT
    normalize = weights.transforms()
    train_tf = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize.mean, std=normalize.std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize.mean, std=normalize.std),
        ]
    )
    return train_tf, eval_tf


def _loader(rows: list[DatasetRow], train: bool) -> DataLoader:
    train_tf, eval_tf = _make_transforms()
    dataset = TileDataset(rows, train_tf if train else eval_tf)
    return DataLoader(dataset, batch_size=32, shuffle=train, num_workers=0)


def train_mobilenet(
    run_name: str = "real-v1",
    export_name: str = "real-balanced-256",
    epochs: int = 6,
) -> dict[str, object]:
    labels_csv = repo_root() / "data" / "processed" / run_name / "exports" / export_name / "labels.csv"
    model_root = repo_root() / "models" / run_name / export_name
    model_root.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(labels_csv)
    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "val"]
    test_rows = [row for row in rows if row.split == "test"]

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
