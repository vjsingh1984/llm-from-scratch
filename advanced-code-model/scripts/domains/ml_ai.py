"""
Machine Learning & AI Domain

Covers:
- PyTorch & TensorFlow
- Scikit-learn
- Hugging Face Transformers
- MLOps (MLflow, Weights & Biases)
- Model training, evaluation, deployment
- Feature engineering
- Neural network architectures
"""

from typing import List
from .base import BaseDomain, DomainExample


class MLAIDomain(BaseDomain):
    """Machine Learning and AI training examples."""

    def get_name(self) -> str:
        return "Machine Learning & AI"

    def get_description(self) -> str:
        return "PyTorch, TensorFlow, scikit-learn, Transformers, MLOps, and deep learning"

    def get_subdomains(self) -> List[str]:
        return [
            "pytorch", "tensorflow", "sklearn", "transformers",
            "mlops", "feature_engineering", "deployment"
        ]

    def get_examples(self) -> List[DomainExample]:
        examples = []
        examples.extend(self._pytorch_examples())
        examples.extend(self._tensorflow_examples())
        examples.extend(self._sklearn_examples())
        examples.extend(self._transformers_examples())
        examples.extend(self._mlops_examples())
        examples.extend(self._feature_engineering_examples())
        return examples

    def _pytorch_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create a PyTorch neural network for image classification with training loop",
                code='''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Define CNN architecture
class ImageClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


def main():
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 50
    batch_size = 128
    learning_rate = 0.001

    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # Datasets
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model, criterion, optimizer
    model = ImageClassifier(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
              f"Test Loss={test_loss:.4f}, Acc={test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")

    print(f"Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()''',
                domain="ml_ai",
                subdomain="pytorch",
                tags=["cnn", "classification", "training"],
                difficulty="intermediate"
            ),
            DomainExample(
                prompt="Implement a PyTorch Transformer model from scratch",
                code='''import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape

        # Project and reshape to (batch, heads, seq, head_dim)
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture
        attn_out = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

        # Weight tying
        self.output.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return self.output(x)

    @staticmethod
    def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0''',
                domain="ml_ai",
                subdomain="pytorch",
                tags=["transformer", "attention", "architecture"],
                difficulty="advanced"
            ),
            DomainExample(
                prompt="Create PyTorch custom dataset and data augmentation pipeline",
                code='''import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import pandas as pd
from pathlib import Path
import random
from typing import Tuple, Optional, Callable, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class CustomImageDataset(Dataset):
    """Custom dataset for image classification with augmentations."""

    def __init__(
        self,
        data_dir: str,
        csv_file: str,
        transform: Optional[Callable] = None,
        is_training: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.is_training = is_training

        # Create label mapping
        self.classes = sorted(self.df["label"].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = self.data_dir / row["image_path"]
        label = self.class_to_idx[row["label"]]

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Training augmentations using Albumentations."""
    return A.Compose([
        A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.05),
            A.GridDistortion(distort_limit=0.05),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
        ], p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=4.0),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Validation transforms (no augmentation)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class MixupCutmix:
    """Mixup and Cutmix augmentation for batches."""

    def __init__(self, mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0, prob: float = 0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if random.random() > self.prob:
            return images, labels, labels, 1.0

        batch_size = images.size(0)
        indices = torch.randperm(batch_size)

        if random.random() > 0.5:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            mixed_images = lam * images + (1 - lam) * images[indices]
        else:
            # Cutmix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
            mixed_images = images.clone()
            mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))

        return mixed_images, labels, labels[indices], lam

    def _rand_bbox(self, size, lam):
        W, H = size[2], size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)

        cx, cy = np.random.randint(W), np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


def create_dataloaders(
    data_dir: str,
    train_csv: str,
    val_csv: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""

    train_dataset = CustomImageDataset(
        data_dir=data_dir,
        csv_file=train_csv,
        transform=get_train_transforms(image_size),
        is_training=True
    )

    val_dataset = CustomImageDataset(
        data_dir=data_dir,
        csv_file=val_csv,
        transform=get_val_transforms(image_size),
        is_training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader''',
                domain="ml_ai",
                subdomain="pytorch",
                tags=["dataset", "augmentation", "dataloader"],
                difficulty="intermediate"
            ),
        ]

    def _tensorflow_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create a TensorFlow Keras model with custom training loop",
                code='''import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class ResidualBlock(layers.Layer):
    """Residual block with pre-activation."""

    def __init__(self, filters: int, strides: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.strides = strides

    def build(self, input_shape):
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(self.filters, 3, strides=self.strides, padding="same")
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(self.filters, 3, padding="same")

        if self.strides > 1 or input_shape[-1] != self.filters:
            self.shortcut = layers.Conv2D(self.filters, 1, strides=self.strides)
        else:
            self.shortcut = lambda x: x

    def call(self, inputs, training=False):
        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        return x + self.shortcut(inputs)


def create_resnet(input_shape=(32, 32, 3), num_classes=10):
    """Create ResNet-like model."""
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(64, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Residual blocks
    for filters, blocks, strides in [(64, 2, 1), (128, 2, 2), (256, 2, 2)]:
        for i in range(blocks):
            x = ResidualBlock(filters, strides if i == 0 else 1)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes)(x)

    return keras.Model(inputs, outputs)


class Trainer:
    """Custom training loop with mixed precision and gradient accumulation."""

    def __init__(self, model, optimizer, loss_fn, accumulation_steps=1):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.accumulation_steps = accumulation_steps

        # Metrics
        self.train_loss = keras.metrics.Mean()
        self.train_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.val_loss = keras.metrics.Mean()
        self.val_accuracy = keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.loss_fn(y, logits)
            scaled_loss = loss / self.accumulation_steps

        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        return gradients, loss, logits

    def train_epoch(self, train_dataset, val_dataset):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()

        accumulated_gradients = [tf.zeros_like(v) for v in self.model.trainable_variables]

        for step, (x, y) in enumerate(train_dataset):
            gradients, loss, logits = self.train_step(x, y)

            # Accumulate gradients
            for i, grad in enumerate(gradients):
                if grad is not None:
                    accumulated_gradients[i] += grad

            # Apply gradients after accumulation
            if (step + 1) % self.accumulation_steps == 0:
                self.optimizer.apply_gradients(
                    zip(accumulated_gradients, self.model.trainable_variables)
                )
                accumulated_gradients = [tf.zeros_like(v) for v in self.model.trainable_variables]

            self.train_loss.update_state(loss)
            self.train_accuracy.update_state(y, logits)

        # Validation
        self.val_loss.reset_states()
        self.val_accuracy.reset_states()

        for x, y in val_dataset:
            logits = self.model(x, training=False)
            loss = self.loss_fn(y, logits)
            self.val_loss.update_state(loss)
            self.val_accuracy.update_state(y, logits)

        return {
            "train_loss": self.train_loss.result().numpy(),
            "train_acc": self.train_accuracy.result().numpy(),
            "val_loss": self.val_loss.result().numpy(),
            "val_acc": self.val_accuracy.result().numpy(),
        }


def main():
    # Enable mixed precision
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Create datasets with augmentation
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_ds = val_ds.batch(128).prefetch(tf.data.AUTOTUNE)

    # Create model and trainer
    model = create_resnet(num_classes=10)
    optimizer = keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=0.01)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    trainer = Trainer(model, optimizer, loss_fn, accumulation_steps=2)

    # Training loop
    for epoch in range(100):
        metrics = trainer.train_epoch(train_ds, val_ds)
        print(f"Epoch {epoch+1}: {metrics}")


if __name__ == "__main__":
    main()''',
                domain="ml_ai",
                subdomain="tensorflow",
                tags=["keras", "custom_training", "mixed_precision"],
                difficulty="advanced"
            ),
        ]

    def _sklearn_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Build a complete ML pipeline with scikit-learn for classification",
                code='''import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.feature_selection import SelectFromModel, RFE
import joblib
import warnings
warnings.filterwarnings("ignore")


def create_preprocessing_pipeline(numeric_features: list, categorical_features: list):
    """Create preprocessing pipeline for mixed data types."""

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor


def create_model_pipeline(preprocessor, model):
    """Create complete pipeline with preprocessing and model."""
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])


def evaluate_models(X, y, preprocessor, models: dict, cv: int = 5):
    """Evaluate multiple models using cross-validation."""
    results = {}

    for name, model in models.items():
        pipeline = create_model_pipeline(preprocessor, model)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
        results[name] = {
            "mean_auc": scores.mean(),
            "std_auc": scores.std(),
            "scores": scores
        }
        print(f"{name}: AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")

    return results


def hyperparameter_tuning(X_train, y_train, preprocessor, param_grid: dict):
    """Perform hyperparameter tuning with GridSearchCV."""

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # Add classifier prefix to param names
    param_grid_pipeline = {f"classifier__{k}": v for k, v in param_grid.items()}

    grid_search = GridSearchCV(
        pipeline,
        param_grid_pipeline,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best AUC: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def main():
    # Load data
    df = pd.read_csv("data.csv")

    # Define features
    numeric_features = ["age", "income", "credit_score", "account_age"]
    categorical_features = ["gender", "education", "occupation", "region"]
    target = "churn"

    X = df[numeric_features + categorical_features]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create preprocessor
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)

    # Define models to evaluate
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }

    # Evaluate models
    print("Model Evaluation:")
    results = evaluate_models(X_train, y_train, preprocessor, models)

    # Hyperparameter tuning for best model
    print("\\nHyperparameter Tuning:")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    best_model = hyperparameter_tuning(X_train, y_train, preprocessor, param_grid)

    # Final evaluation
    print("\\nFinal Evaluation on Test Set:")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Save model
    joblib.dump(best_model, "best_model.joblib")
    print("\\nModel saved to best_model.joblib")


if __name__ == "__main__":
    main()''',
                domain="ml_ai",
                subdomain="sklearn",
                tags=["pipeline", "classification", "hyperparameter_tuning"],
                difficulty="intermediate"
            ),
        ]

    def _transformers_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Fine-tune a Hugging Face transformer model for text classification",
                code='''import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from typing import Dict
import wandb


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize text examples."""
    return tokenizer(
        examples["text"],
        padding=False,
        truncation=True,
        max_length=max_length
    )


def main():
    # Initialize wandb
    wandb.init(project="text-classification", name="bert-sentiment")

    # Config
    model_name = "bert-base-uncased"
    num_labels = 3
    max_length = 256
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 5

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )

    # Load and prepare dataset
    dataset = load_dataset("csv", data_files={
        "train": "train.csv",
        "validation": "val.csv",
        "test": "test.csv"
    })

    # Tokenize
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing"
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=learning_rate,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
        report_to="wandb",
        push_to_hub=False,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train
    trainer.train()

    # Evaluate on test set
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print(f"Test Results: {test_results}")

    # Save model
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")

    wandb.finish()


if __name__ == "__main__":
    main()''',
                domain="ml_ai",
                subdomain="transformers",
                tags=["bert", "fine_tuning", "classification"],
                difficulty="intermediate"
            ),
            DomainExample(
                prompt="Implement RAG (Retrieval Augmented Generation) with Hugging Face",
                code='''import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    pipeline
)
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class Document:
    """Document for RAG."""
    id: str
    content: str
    metadata: dict = None


class RAGSystem:
    """Retrieval Augmented Generation system."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generation_model: str = "microsoft/phi-2",
        index_path: Optional[str] = None
    ):
        # Embedding model for retrieval
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()

        # Generation model
        self.tokenizer = AutoTokenizer.from_pretrained(generation_model)
        self.generator = AutoModelForCausalLM.from_pretrained(
            generation_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # FAISS index
        self.index = None
        self.documents: List[Document] = []

        if index_path:
            self.load_index(index_path)

    def create_index(self, documents: List[Document]):
        """Create FAISS index from documents."""
        self.documents = documents
        texts = [doc.content for doc in documents]

        # Compute embeddings
        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        self.index.add(embeddings.astype(np.float32))

        print(f"Indexed {len(documents)} documents")

    def save_index(self, path: str):
        """Save index and documents to disk."""
        faiss.write_index(self.index, f"{path}/index.faiss")

        docs_data = [
            {"id": doc.id, "content": doc.content, "metadata": doc.metadata}
            for doc in self.documents
        ]
        with open(f"{path}/documents.json", "w") as f:
            json.dump(docs_data, f)

    def load_index(self, path: str):
        """Load index and documents from disk."""
        self.index = faiss.read_index(f"{path}/index.faiss")

        with open(f"{path}/documents.json", "r") as f:
            docs_data = json.load(f)

        self.documents = [
            Document(id=d["id"], content=d["content"], metadata=d.get("metadata"))
            for d in docs_data
        ]

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve top-k relevant documents."""
        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = self.index.search(query_embedding.astype(np.float32), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))

        return results

    def generate(
        self,
        query: str,
        context_docs: List[Document],
        max_new_tokens: int = 256
    ) -> str:
        """Generate answer using retrieved context."""

        # Build context
        context = "\\n\\n".join([
            f"Document {i+1}:\\n{doc.content}"
            for i, doc in enumerate(context_docs)
        ])

        # Build prompt
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.generator.device)

        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer (after "Answer:")
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()

        return response

    def query(self, question: str, k: int = 5, max_new_tokens: int = 256) -> dict:
        """Complete RAG pipeline: retrieve and generate."""

        # Retrieve
        retrieved = self.retrieve(question, k=k)
        context_docs = [doc for doc, _ in retrieved]

        # Generate
        answer = self.generate(question, context_docs, max_new_tokens)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"id": doc.id, "score": score, "content": doc.content[:200] + "..."}
                for doc, score in retrieved
            ]
        }


def main():
    # Example usage
    rag = RAGSystem()

    # Create sample documents
    documents = [
        Document(id="1", content="Python is a programming language known for its simplicity."),
        Document(id="2", content="Machine learning is a subset of artificial intelligence."),
        Document(id="3", content="PyTorch is a deep learning framework developed by Meta."),
    ]

    rag.create_index(documents)

    # Query
    result = rag.query("What is PyTorch?")
    print(f"Question: {result[\'question\']}")
    print(f"Answer: {result[\'answer\']}")
    print(f"Sources: {result[\'sources\']}")


if __name__ == "__main__":
    main()''',
                domain="ml_ai",
                subdomain="transformers",
                tags=["rag", "retrieval", "generation"],
                difficulty="advanced"
            ),
        ]

    def _mlops_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Set up MLflow experiment tracking and model registry",
                code='''import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
from typing import Dict, Any
import os


def setup_mlflow(
    tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "my-experiment"
):
    """Configure MLflow tracking."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Enable autologging for common frameworks
    mlflow.sklearn.autolog(log_models=True)
    mlflow.pytorch.autolog(log_models=True)


def log_sklearn_model(
    model,
    X_train, X_test, y_train, y_test,
    params: Dict[str, Any],
    model_name: str = "model"
):
    """Train and log sklearn model with MLflow."""

    with mlflow.start_run(run_name=model_name) as run:
        # Log parameters
        mlflow.log_params(params)

        # Train
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        mlflow.log_metrics(metrics)

        # Log model
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=model_name
        )

        # Log feature importance
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "feature": X_train.columns,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
            importance_df.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")

        print(f"Run ID: {run.info.run_id}")
        print(f"Metrics: {metrics}")

        return run.info.run_id


def register_and_promote_model(
    run_id: str,
    model_name: str,
    stage: str = "Staging"
):
    """Register model and promote to a stage."""
    client = MlflowClient()

    # Register model
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)

    # Wait for registration
    from mlflow.tracking.client import MlflowClient
    client = MlflowClient()

    # Promote to stage
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage=stage
    )

    print(f"Model {model_name} v{result.version} promoted to {stage}")

    return result.version


def load_production_model(model_name: str):
    """Load model from production stage."""
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.sklearn.load_model(model_uri)
    return model


class PyTorchModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper for PyTorch models in MLflow."""

    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, context, model_input):
        self.model.eval()
        with torch.no_grad():
            if self.tokenizer:
                inputs = self.tokenizer(
                    model_input["text"].tolist(),
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                outputs = self.model(**inputs)
            else:
                inputs = torch.tensor(model_input.values)
                outputs = self.model(inputs)

        return outputs.cpu().numpy()


def log_pytorch_model(model, sample_input, model_name: str):
    """Log PyTorch model with MLflow."""

    with mlflow.start_run(run_name=f"pytorch-{model_name}"):
        # Log model architecture
        mlflow.log_text(str(model), "model_architecture.txt")

        # Log model
        signature = infer_signature(
            sample_input.numpy(),
            model(sample_input).detach().numpy()
        )

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=model_name
        )


def compare_runs(experiment_name: str, metric: str = "accuracy"):
    """Compare runs in an experiment."""
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"]
    )

    comparison = []
    for run in runs:
        comparison.append({
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", ""),
            metric: run.data.metrics.get(metric, 0),
            "params": run.data.params
        })

    return pd.DataFrame(comparison)


if __name__ == "__main__":
    # Setup
    setup_mlflow(experiment_name="classification-experiment")

    # Load data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train and log model
    params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    model = RandomForestClassifier(**params)

    run_id = log_sklearn_model(
        model, X_train, X_test, y_train, y_test,
        params=params,
        model_name="random-forest-classifier"
    )

    # Register and promote
    version = register_and_promote_model(
        run_id,
        "random-forest-classifier",
        stage="Staging"
    )''',
                domain="ml_ai",
                subdomain="mlops",
                tags=["mlflow", "experiment_tracking", "model_registry"],
                difficulty="intermediate"
            ),
        ]

    def _feature_engineering_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create a feature engineering pipeline with various transformations",
                code='''import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, PowerTransformer,
    PolynomialFeatures, KBinsDiscretizer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from category_encoders import TargetEncoder, WOEEncoder
import warnings
warnings.filterwarnings("ignore")


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from datetime columns."""

    def __init__(self, date_columns):
        self.date_columns = date_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.date_columns:
            X[col] = pd.to_datetime(X[col])
            X[f"{col}_year"] = X[col].dt.year
            X[f"{col}_month"] = X[col].dt.month
            X[f"{col}_day"] = X[col].dt.day
            X[f"{col}_dayofweek"] = X[col].dt.dayofweek
            X[f"{col}_is_weekend"] = X[col].dt.dayofweek >= 5
            X[f"{col}_quarter"] = X[col].dt.quarter
            X[f"{col}_is_month_start"] = X[col].dt.is_month_start
            X[f"{col}_is_month_end"] = X[col].dt.is_month_end

            # Cyclical encoding for month and day of week
            X[f"{col}_month_sin"] = np.sin(2 * np.pi * X[col].dt.month / 12)
            X[f"{col}_month_cos"] = np.cos(2 * np.pi * X[col].dt.month / 12)
            X[f"{col}_dow_sin"] = np.sin(2 * np.pi * X[col].dt.dayofweek / 7)
            X[f"{col}_dow_cos"] = np.cos(2 * np.pi * X[col].dt.dayofweek / 7)

            X = X.drop(col, axis=1)

        return X


class AggregateFeatures(BaseEstimator, TransformerMixin):
    """Create aggregate features based on grouping."""

    def __init__(self, group_cols, agg_cols, agg_funcs=["mean", "std", "min", "max"]):
        self.group_cols = group_cols
        self.agg_cols = agg_cols
        self.agg_funcs = agg_funcs
        self.agg_features_ = None

    def fit(self, X, y=None):
        # Compute aggregations on training data
        self.agg_features_ = X.groupby(self.group_cols)[self.agg_cols].agg(self.agg_funcs)
        self.agg_features_.columns = [
            f"{col}_{func}_by_{\'_\'.join(self.group_cols)}"
            for col, func in self.agg_features_.columns
        ]
        return self

    def transform(self, X):
        X = X.copy()
        X = X.merge(self.agg_features_, on=self.group_cols, how="left")
        return X


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Handle outliers using IQR or Z-score method."""

    def __init__(self, method="iqr", factor=1.5, columns=None):
        self.method = method
        self.factor = factor
        self.columns = columns
        self.bounds_ = {}

    def fit(self, X, y=None):
        cols = self.columns or X.select_dtypes(include=[np.number]).columns

        for col in cols:
            if self.method == "iqr":
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.bounds_[col] = (Q1 - self.factor * IQR, Q3 + self.factor * IQR)
            elif self.method == "zscore":
                mean = X[col].mean()
                std = X[col].std()
                self.bounds_[col] = (mean - self.factor * std, mean + self.factor * std)

        return self

    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            X[col] = X[col].clip(lower=lower, upper=upper)
        return X


class InteractionFeatures(BaseEstimator, TransformerMixin):
    """Create interaction features between numeric columns."""

    def __init__(self, columns=None, operations=["multiply", "divide", "add", "subtract"]):
        self.columns = columns
        self.operations = operations

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols = self.columns or X.select_dtypes(include=[np.number]).columns.tolist()

        for i, col1 in enumerate(cols):
            for col2 in cols[i+1:]:
                if "multiply" in self.operations:
                    X[f"{col1}_x_{col2}"] = X[col1] * X[col2]
                if "divide" in self.operations:
                    X[f"{col1}_div_{col2}"] = X[col1] / (X[col2] + 1e-8)
                if "add" in self.operations:
                    X[f"{col1}_plus_{col2}"] = X[col1] + X[col2]
                if "subtract" in self.operations:
                    X[f"{col1}_minus_{col2}"] = X[col1] - X[col2]

        return X


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical variables by their frequency."""

    def __init__(self, columns=None):
        self.columns = columns
        self.freq_maps_ = {}

    def fit(self, X, y=None):
        cols = self.columns or X.select_dtypes(include=["object", "category"]).columns

        for col in cols:
            self.freq_maps_[col] = X[col].value_counts(normalize=True).to_dict()

        return self

    def transform(self, X):
        X = X.copy()
        for col, freq_map in self.freq_maps_.items():
            X[f"{col}_freq"] = X[col].map(freq_map).fillna(0)
        return X


def create_feature_pipeline(
    numeric_cols,
    categorical_cols,
    date_cols,
    target_encode_cols=None
):
    """Create comprehensive feature engineering pipeline."""

    # Numeric pipeline
    numeric_pipeline = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("outlier", OutlierHandler(method="iqr", factor=1.5)),
        ("scaler", StandardScaler()),
        ("interactions", InteractionFeatures(operations=["multiply", "divide"])),
    ])

    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("freq_encoder", FrequencyEncoder()),
    ])

    # Date pipeline
    date_pipeline = Pipeline([
        ("extractor", DateFeatureExtractor(date_cols))
    ])

    # Combine all
    full_pipeline = Pipeline([
        ("date_features", date_pipeline),
        ("feature_union", FeatureUnion([
            ("numeric", numeric_pipeline),
            ("categorical", categorical_pipeline),
        ])),
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("pca", PCA(n_components=0.95))  # Keep 95% variance
    ])

    return full_pipeline


if __name__ == "__main__":
    # Example usage
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=100),
        "numeric1": np.random.randn(100),
        "numeric2": np.random.randn(100) * 10,
        "category": np.random.choice(["A", "B", "C"], 100)
    })

    pipeline = create_feature_pipeline(
        numeric_cols=["numeric1", "numeric2"],
        categorical_cols=["category"],
        date_cols=["date"]
    )

    transformed = pipeline.fit_transform(df)
    print(f"Original shape: {df.shape}")
    print(f"Transformed shape: {transformed.shape}")''',
                domain="ml_ai",
                subdomain="feature_engineering",
                tags=["pipeline", "transformers", "preprocessing"],
                difficulty="advanced"
            ),
        ]
