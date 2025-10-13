with these 3 methods: complex moments, zernike moments, and fourier mellin moments, I want to train 3 models: baseline, early fusion, and late fusion

i want to train them on a subset of 50, 100, 150, 200, 300, 500 images and then make a comprehensive table containing the moments type, the models and the subset size of data. write a code that does all this:






import os
import numpy as np
import cv2
import mahotas
from tqdm.notebook import tqdm
import shutil

# --- Config ---
BASE_INPUT_PATH = '/kaggle/input/sarscov2-ctscan-dataset' 
CATEGORIES = ['COVID', 'non-COVID']
FEATURES_PATH = '/kaggle/working/features_zernike/' 

IMG_WIDTH = 128
IMG_HEIGHT = 128
SUBSET_SIZE_PER_CATEGORY = 50
ZERNIKE_RADIUS = 64   # Rayon du cercle pour les moments (la moitié de 128)

# --- Fonction : calcul des invariants de Zernike ---
def get_zernike_features(image, radius=ZERNIKE_RADIUS, degree=8):
    """
    Calcule les moments invariants de Zernike pour une image.
    - image : np.array RGB ou grayscale
    - radius : rayon du cercle inscriptible
    - degree : ordre maximum des moments
    Retourne un vecteur de features 1D.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    gray = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
    return mahotas.features.zernike_moments(gray, radius, degree)

# --- Main ---
if __name__ == "__main__":
    if os.path.exists(FEATURES_PATH):
        print(f"Removing existing features directory: {FEATURES_PATH}")
        shutil.rmtree(FEATURES_PATH)
    
    print("Creating new feature directories for Zernike moments...")
    for category in CATEGORIES:
        os.makedirs(os.path.join(FEATURES_PATH, category), exist_ok=True)

    # Pass 1 : Calcul des features bruts
    print("\n--- Pass 1: Calcul des Zernike moments ---")
    normalization_stats = []
    feature_cache = {}

    for category in CATEGORIES:
        print(f"\nProcessing category: {category}")
        category_path = os.path.join(BASE_INPUT_PATH, category)
        
        if not os.path.isdir(category_path):
            print(f"Warning: Directory not found for '{category}'. Skipping.")
            continue
            
        image_files = sorted(os.listdir(category_path))
        if SUBSET_SIZE_PER_CATEGORY is not None:
            image_files = image_files[:SUBSET_SIZE_PER_CATEGORY]

        for filename in tqdm(image_files, desc=f"Pass 1/2 - {category}"):
            image_path = os.path.join(category_path, filename)
            image = cv2.imread(image_path)
            if image is None: 
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Calcul des invariants de Zernike
            features = get_zernike_features(image, radius=ZERNIKE_RADIUS, degree=8)
            
            feature_cache[(category, filename)] = features
            normalization_stats.append(features)

    # Stats globales pour normalisation
    all_features = np.vstack(list(feature_cache.values()))
    min_vals = all_features.min(axis=0)
    max_vals = all_features.max(axis=0)

    print("\n--- Normalization Stats ---")
    print(f"  Shape of feature vector: {all_features.shape[1]}")
    print(f"  Min (first 5): {min_vals[:5]}")
    print(f"  Max (first 5): {max_vals[:5]}")

    # Pass 2 : Normalisation + sauvegarde
    print("\n--- Pass 2: Normalizing and saving features ---")
    epsilon = 1e-9
    for (category, filename), features in tqdm(feature_cache.items(), desc="Pass 2/2 - Saving"):
        norm_features = (features - min_vals) / (max_vals - min_vals + epsilon)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(FEATURES_PATH, category, f"{base_name}.npy")
        np.save(output_path, norm_features)

    print("\nFeature pre-computation with Zernike invariants is complete.")



import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Paths and Parameters for the new dataset
BASE_DATA_PATH = '/kaggle/input/sarscov2-ctscan-dataset/'
# This path MUST point to the output of your updated 'calculate_features' script
FEATURES_PATH = '/kaggle/working/features_zernike/'
CATEGORIES = ['COVID', 'non-COVID']
CLASS_TO_IDX = {name: i for i, name in enumerate(CATEGORIES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}


IMG_SIZE = 128
BATCH_SIZE = 25
EPOCHS = 30
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.5


import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- 1. Configuration ---
BASE_DATA_PATH = '/kaggle/input/sarscov2-ctscan-dataset/'
FEATURES_PATH = '/kaggle/working/features_zernike/'
CATEGORIES = ['COVID', 'non-COVID']
CLASS_TO_IDX = {name: i for i, name in enumerate(CATEGORIES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}

num_samples_per_category = 2  # Nombre d’images à afficher par catégorie

# --- 2. Helper function pour trouver les images originales ---
def find_image_path(base_path, category, base_filename):
    for ext in ['.png', '.jpg', '.jpeg', '.tif']:
        img_path = os.path.join(base_path, category, base_filename + ext)
        if os.path.exists(img_path):
            return img_path
    return None

# --- 3. Collecte des échantillons ---
samples_to_plot = []

for category in CATEGORIES:
    feature_category_path = os.path.join(FEATURES_PATH, category)
    if not os.path.isdir(feature_category_path):
        print(f"Warning: Feature directory not found for '{category}'. Skipping.")
        continue

    feature_files = [f for f in os.listdir(feature_category_path) if f.endswith('.npy')]
    if not feature_files:
        print(f"Warning: No feature files found for category '{category}'.")
        continue

    selected_files = random.sample(feature_files, min(num_samples_per_category, len(feature_files)))

    for fname in selected_files:
        base_name = os.path.splitext(fname)[0]
        img_path = find_image_path(BASE_DATA_PATH, category, base_name)
        feature_path = os.path.join(feature_category_path, fname)
        if img_path and os.path.exists(feature_path):
            samples_to_plot.append({
                'category': category,
                'original': img_path,
                'feature': feature_path
            })

# --- 4. Visualisation ---
if not samples_to_plot:
    print("No samples to display. Make sure the features exist and paths are correct.")
else:
    for sample in samples_to_plot:
        original_img = Image.open(sample['original']).convert('RGB')
        zernike_vector = np.load(sample['feature'])

        plt.figure(figsize=(10,4))

        # --- Image originale ---
        plt.subplot(1,2,1)
        plt.imshow(original_img, cmap='gray')
        plt.title(f"Original Image ({sample['category']})")
        plt.axis('off')

        # --- Vecteur Zernike ---
        plt.subplot(1,2,2)
        plt.bar(range(len(zernike_vector)), zernike_vector)
        plt.title("Zernike Feature Vector")
        plt.xlabel("Feature Index")
        plt.ylabel("Value")

        plt.tight_layout()
        plt.show()


# --- 1. Dataset ---
class FusionDataset(Dataset):
    def __init__(self, file_paths, labels, features_dir, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.features_dir = features_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        # Charger image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image_tensor = self.transform(image)

        # Charger vecteur Zernike
        category = CATEGORIES[label]
        base_name = os.path.splitext(os.path.basename(img_path))[0] + '.npy'
        zernike_path = os.path.join(self.features_dir, category, base_name)
        zernike_vector = torch.from_numpy(np.load(zernike_path)).float()

        return image_tensor, zernike_vector, torch.tensor(label, dtype=torch.long)


class BaselineVGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        vgg = models.vgg16(weights=None)
        vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, num_classes)
        self.vgg = vgg
    def forward(self, x_image, x_zernike=None):
        return self.vgg(x_image)



class EarlyFusionVGG16(nn.Module):
    def __init__(self, num_classes, zernike_dim, img_size=IMG_SIZE):
        super().__init__()
        vgg_base = models.vgg16(weights=None)
        original_first_layer = vgg_base.features[0]
        self.features_first_conv = nn.Conv2d(3 + zernike_dim, 64, kernel_size=3, padding=1)
        self.features_first_conv.weight.data[:, :3, :, :] = original_first_layer.weight.data
        self.features_first_conv.bias.data = original_first_layer.bias.data
        nn.init.kaiming_normal_(self.features_first_conv.weight.data[:, 3:, :, :])
        self.features_rest = vgg_base.features[1:]
        self.avgpool = vgg_base.avgpool
        self.classifier = vgg_base.classifier
        self.classifier[6] = nn.Linear(self.classifier[6].in_features, num_classes)
        self.img_size = img_size
        self.zernike_dim = zernike_dim
    def forward(self, x_image, x_zernike):
        B = x_image.size(0)
        zernike_maps = x_zernike.unsqueeze(-1).unsqueeze(-1)
        zernike_maps = zernike_maps.expand(-1, -1, self.img_size, self.img_size)
        x_combined = torch.cat([x_image, zernike_maps], dim=1)
        x = self.features_first_conv(x_combined)
        x = self.features_rest(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class LateFusionVGG16(nn.Module):
    def __init__(self, num_classes, zernike_dim):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.cnn_features = vgg.features
        self.avgpool = vgg.avgpool
        self.cnn_fc = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(True)
        )
        self.zernike_fc = nn.Sequential(
            nn.Linear(zernike_dim, 256),
            nn.ReLU(True)
        )
        self.classifier = nn.Linear(1024 + 256, num_classes)
    def forward(self, x_image, x_zernike):
        x = self.cnn_features(x_image)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.cnn_fc(x)
        z = self.zernike_fc(x_zernike)
        combined = torch.cat([x, z], dim=1)
        return self.classifier(combined)

# --- 2. Helper function pour matcher les images ---
def find_original_image(base_path, category, base_filename):
    category_path = os.path.join(base_path, category)
    if not os.path.isdir(category_path):
        return None
    for file in os.listdir(category_path):
        name, ext = os.path.splitext(file)
        if name.lower() == base_filename.lower() and ext.lower() in ['.png', '.jpg', '.jpeg', '.tif']:
            return os.path.join(category_path, file)
    return None

# --- 3. Scan des fichiers et labels ---
all_files, all_labels = [], []

for category in CATEGORIES:
    feature_category_path = os.path.join(FEATURES_PATH, category)
    if not os.path.isdir(feature_category_path):
        print(f"Warning: Feature directory not found for '{category}'. Skipping.")
        continue

    for feature_filename in os.listdir(feature_category_path):
        if feature_filename.endswith('.npy'):
            base_name = os.path.splitext(feature_filename)[0]
            original_image_path = find_original_image(BASE_DATA_PATH, category, base_name)
            if original_image_path:
                all_files.append(original_image_path)
                all_labels.append(CLASS_TO_IDX[category])
            else:
                print(f"Warning: Feature file '{feature_filename}' exists, but no matching image found for '{base_name}'")

if not all_files:
    raise ValueError("Aucun fichier Zernike trouvé ou aucune image correspondante.")

# Split train/val
train_files, val_files, train_labels, val_labels = train_test_split(
    all_files, all_labels, test_size=VALIDATION_SPLIT, random_state=42, stratify=all_labels
)

# --- 4. Transformations et DataLoaders ---
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = FusionDataset(train_files, train_labels, FEATURES_PATH, transform=data_transforms)
val_dataset   = FusionDataset(val_files, val_labels, FEATURES_PATH, transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

zernike_dim = train_dataset[0][1].shape[0]




# --- 6. Boucle d’entraînement ---
def train_and_validate(model_name, model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0
        for imgs, zernike_vecs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            imgs, zernike_vecs, labels = imgs.to(device), zernike_vecs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs, zernike_vecs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            running_corrects += (outputs.argmax(1) == labels).sum().item()
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects / len(train_loader.dataset)
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for imgs, zernike_vecs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                imgs, zernike_vecs, labels = imgs.to(device), zernike_vecs.to(device), labels.to(device)
                outputs = model(imgs, zernike_vecs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_corrects += (outputs.argmax(1) == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects / len(val_loader.dataset)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"[{model_name}] Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    return history
print("--- Data Partitioning Summary ---")
print(f"Total images with features: {len(all_files)}")
print(f"Training set size:          {len(train_dataset)} images")
print(f"Validation set size:        {len(val_dataset)} images")
print("-" * 33)

# --- 7. Main Execution ---
models_to_run = {
    'Baseline': BaselineVGG16(num_classes=len(CATEGORIES)),
    'Early Fusion': EarlyFusionVGG16(num_classes=len(CATEGORIES), zernike_dim=zernike_dim),
    'Late Fusion': LateFusionVGG16(num_classes=len(CATEGORIES), zernike_dim=zernike_dim)
}

all_histories = {}
for name, model in models_to_run.items():
    print(f"\n{'='*50}\nTraining Model: {name}\n{'='*50}")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    history = train_and_validate(
        model_name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=EPOCHS,
        device=DEVICE
    )
    all_histories[name] = history


import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

# --- 1. Training History over Epochs ---
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Training and Validation History', fontsize=16)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for color, (name, history) in zip(colors, all_histories.items()):
    axes[0].plot(history['val_acc'], 'o--', color=color, label=f'{name} Val Acc')
    axes[1].plot(history['val_loss'], 'x--', color=color, label=f'{name} Val Loss')

# Accuracy plot
axes[0].set_title('Validation Accuracy over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(0, 1.0)
axes[0].legend()
axes[0].grid(True)

# Loss plot
axes[1].set_title('Validation Loss over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# --- 2. Final Validation Performance ---
final_val_acc = {name: h['val_acc'][-1] for name, h in all_histories.items()}
final_val_loss = {name: h['val_loss'][-1] for name, h in all_histories.items()}

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Final Validation Performance Comparison', fontsize=16)

# Accuracy bar chart
bars_acc = axes[0].bar(final_val_acc.keys(), final_val_acc.values(), color=colors, edgecolor='black')
axes[0].set_title('Final Validation Accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(0, 1.0)
axes[0].bar_label(bars_acc, fmt='{:.2%}')

# Loss bar chart
bars_loss = axes[1].bar(final_val_loss.keys(), final_val_loss.values(), color=colors, edgecolor='black')
axes[1].set_title('Final Validation Loss')
axes[1].set_ylabel('Loss')
axes[1].bar_label(bars_loss, fmt='{:.4f}')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



import os
import numpy as np
import cv2
from tqdm.notebook import tqdm
import shutil

# --- Configuration ---
BASE_INPUT_PATH = '/kaggle/input/sarscov2-ctscan-dataset/' 
CATEGORIES = ['COVID', 'non-COVID']
FEATURES_PATH = '/kaggle/working/features_fm_128/' 
SUBSET_SIZE_PER_CATEGORY = 300
IMG_WIDTH = 128
IMG_HEIGHT = 128

# --- Fourier-Mellin Feature Functions ---

def fourier_mellin_feature(image):
    """
    Compute Fourier-Mellin invariant feature (magnitude spectrum of log-polar transform)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))

    # FFT and shift
    f = np.fft.fft2(resized)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    # Log-polar transform
    center = (IMG_WIDTH // 2, IMG_HEIGHT // 2)
    log_polar = cv2.logPolar(magnitude_spectrum, center, M=40,
                             flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

    # FFT of log-polar image
    f_lp = np.fft.fft2(log_polar)
    f_lp_shift = np.fft.fftshift(f_lp)
    magnitude_lp = np.abs(f_lp_shift)
    # Normalize to 0-1
    magnitude_lp = magnitude_lp / (np.max(magnitude_lp) + 1e-9)

    return magnitude_lp.astype(np.float32)

# --- Main Execution ---
if __name__ == "__main__":
    if os.path.exists(FEATURES_PATH):
        shutil.rmtree(FEATURES_PATH)
    for category in CATEGORIES:
        os.makedirs(os.path.join(FEATURES_PATH, category), exist_ok=True)

    for category in CATEGORIES:
        category_path = os.path.join(BASE_INPUT_PATH, category)
        if not os.path.isdir(category_path):
            continue

        image_files = sorted(os.listdir(category_path))
        if SUBSET_SIZE_PER_CATEGORY is not None:
            image_files = image_files[:SUBSET_SIZE_PER_CATEGORY]

        for filename in tqdm(image_files, desc=f"Processing {category}"):
            image_path = os.path.join(category_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            feature_map = fourier_mellin_feature(image)

            output_path = os.path.join(FEATURES_PATH, category, f"{os.path.splitext(filename)[0]}.npy")
            np.save(output_path, feature_map)

    print("Fourier-Mellin feature extraction complete.")



import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Paths and Parameters for the new dataset
BASE_DATA_PATH = '/kaggle/input/sarscov2-ctscan-dataset/'
# This path MUST point to the output of your updated 'calculate_features' script
FEATURES_PATH = '/kaggle/working/features_fm_128/'
CATEGORIES = ['COVID', 'non-COVID']
CLASS_TO_IDX = {name: i for i, name in enumerate(CATEGORIES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}

IMG_SIZE = 128
BATCH_SIZE = 25
EPOCHS = 30
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.5

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- Configuration ---
BASE_DATA_PATH = '/kaggle/input/sarscov2-ctscan-dataset/'
FEATURES_PATH = '/kaggle/working/features_fm_128/'  # Fourier-Mellin features
CATEGORIES = ['COVID', 'non-COVID']

samples_to_plot = []
num_samples_per_category = 2

for category in CATEGORIES:
    image_dir = os.path.join(BASE_DATA_PATH, category)
    fm_dir = os.path.join(FEATURES_PATH, category)

    if not all(os.path.isdir(d) for d in [image_dir, fm_dir]):
        print(f"Warning: Missing directory for '{category}'. Skipping.")
        continue

    feature_files = [f for f in os.listdir(fm_dir) if f.endswith('.npy')]
    if not feature_files:
        print(f"Warning: No Fourier-Mellin features for '{category}'.")
        continue

    selected_files = random.sample(feature_files, min(num_samples_per_category, len(feature_files)))

    for fname in selected_files:
        base_name = os.path.splitext(fname)[0]
        # Find original image (any common extension)
        def find_image(base_path, category, name):
            for ext in ['.png', '.jpg', '.jpeg', '.tif']:
                path = os.path.join(base_path, category, name + ext)
                if os.path.exists(path):
                    return path
            return None

        original_img_path = find_image(BASE_DATA_PATH, category, base_name)
        fm_map_path = os.path.join(fm_dir, fname)

        if original_img_path and os.path.exists(fm_map_path):
            samples_to_plot.append({
                'category': category,
                'original': original_img_path,
                'fm_map': fm_map_path
            })

# --- Plotting ---
if not samples_to_plot:
    print("No matching images/features found.")
else:
    num_samples = len(samples_to_plot)
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, num_samples * 5))
    fig.suptitle('Original Images and their Fourier-Mellin Features', fontsize=16)

    for i, sample in enumerate(samples_to_plot):
        original_img = Image.open(sample['original'])
        fm_map = np.load(sample['fm_map'])

        fm_mean_value = np.mean(fm_map)

        # --- Original Image ---
        ax1 = axes[i, 0] if num_samples > 1 else axes[0]
        ax1.imshow(original_img, cmap='gray')
        ax1.set_title(f"Original Image\nCategory: {sample['category']}")
        ax1.axis('off')

        # --- Fourier-Mellin Feature Map ---
        ax2 = axes[i, 1] if num_samples > 1 else axes[1]
        im = ax2.imshow(fm_map, cmap='viridis', vmin=0, vmax=1)
        ax2.set_title(f"Fourier-Mellin Map\nMean: {fm_mean_value:.4f}")
        ax2.axis('off')
        fig.colorbar(im, ax=ax2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
from torch.utils.data import Dataset
import torch
import os
import numpy as np
from PIL import Image

class FusionDatasetFM(Dataset):
    def __init__(self, file_paths, labels, features_dir, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.features_dir = features_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load and transform the image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.from_numpy(np.array(image)).permute(2,0,1).float() / 255.0

        # Load pre-calculated Fourier-Mellin feature
        base_name = os.path.splitext(os.path.basename(img_path))[0] + '.npy'
        fm_path = os.path.join(self.features_dir, label, base_name)  # assuming features_dir/label/*.npy

        fm_map = torch.from_numpy(np.load(fm_path)).unsqueeze(0)  # (1, H, W)

        # Optionally, use the mean as a scalar feature for late fusion
        feature_vector = torch.tensor([fm_map.mean().item()], dtype=torch.float)

        return image_tensor, fm_map, feature_vector, torch.tensor(label, dtype=torch.long)
class BaselineVGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vgg = models.vgg16(weights=None)
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, num_classes)
    def forward(self, x_image):
        return self.vgg(x_image)


import torch
import torch.nn as nn
from torchvision import models

class EarlyFusionVGG16_FM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        vgg_base = models.vgg16(weights=None)
        original_first_layer = vgg_base.features[0]
        
        # Nouvelle première couche pour 4 canaux (RGB + FM map)
        self.features_first_conv = nn.Conv2d(
            in_channels=4,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        # Copier les poids RGB initiaux et initialiser FM map aléatoirement
        self.features_first_conv.weight.data[:, :3, :, :] = original_first_layer.weight.data
        self.features_first_conv.bias.data = original_first_layer.bias.data
        
        # Reste du backbone VGG16
        self.features_rest = vgg_base.features[1:]
        self.avgpool = vgg_base.avgpool
        self.classifier = vgg_base.classifier
        self.classifier[6] = nn.Linear(self.classifier[6].in_features, num_classes)

    def forward(self, x_image, x_fm):
        # x_image: (B,3,H,W), x_fm: (B,1,H,W)
        x_combined = torch.cat([x_image, x_fm], dim=1)  # (B,4,H,W)
        x = self.features_first_conv(x_combined)
        x = self.features_rest(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
import torch
import torch.nn as nn
from torchvision import models

class LateFusionVGG16_FM(nn.Module):
    def __init__(self, num_classes, num_fm_features=1):
        super().__init__()
        vgg_base = models.vgg16(weights=None)
        self.features = vgg_base.features
        self.avgpool = vgg_base.avgpool

        # Nombre de features d'entrée pour la fusion tardive
        original_in_features = vgg_base.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(original_in_features + num_fm_features, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x_image, x_fm):
        """
        x_image: (B,3,H,W)
        x_fm: (B, num_fm_features)  -> scalar mean or flattened FM map
        """
        x_image = self.features(x_image)
        x_image = self.avgpool(x_image)
        x_image = torch.flatten(x_image, 1)
        x_combined = torch.cat([x_image, x_fm], dim=1)
        output = self.classifier(x_combined)
        return output
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DATA_PATH = '/kaggle/input/sarscov2-ctscan-dataset/'
FEATURES_PATH = '/kaggle/working/features_fm_128/'  # Fourier-Mellin features
CATEGORIES = ['COVID', 'non-COVID']
CLASS_TO_IDX = {name: i for i, name in enumerate(CATEGORIES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}
IMG_SIZE = 128
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2
EPOCHS = 30
LEARNING_RATE = 1e-4

# -----------------------------
# Dataset
# -----------------------------
class FusionDatasetFM(Dataset):
    def __init__(self, file_paths, labels, features_dir, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.features_dir = features_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.from_numpy(np.array(image)).permute(2,0,1).float()/255.0

        # Load Fourier-Mellin feature map
        base_name = os.path.splitext(os.path.basename(img_path))[0] + '.npy'
        fm_path = os.path.join(self.features_dir, IDX_TO_CLASS[label], base_name)
        fm_map = torch.from_numpy(np.load(fm_path)).unsqueeze(0)  # (1,H,W)

        # Feature vector for late fusion (mean of FM map)
        feature_vector = torch.tensor([fm_map.mean().item()], dtype=torch.float)

        return image_tensor, fm_map, feature_vector, torch.tensor(label, dtype=torch.long)

# -----------------------------
# Model Training & Validation
# -----------------------------
def train_and_validate(model_name, model, train_loader, val_loader, criterion, optimizer, num_epochs):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        for images, fm_maps, vectors, labels in train_pbar:
            images, fm_maps, vectors, labels = images.to(DEVICE), fm_maps.to(DEVICE), vectors.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            if model_name == 'Baseline':
                outputs = model(images)
            elif model_name == 'Early Fusion':
                outputs = model(images, fm_maps)
            else:  # Late Fusion
                outputs = model(images, vectors)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        
        # Validation
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, fm_maps, vectors, labels in val_loader:
                images, fm_maps, vectors, labels = images.to(DEVICE), fm_maps.to(DEVICE), vectors.to(DEVICE), labels.to(DEVICE)
                
                if model_name == 'Baseline':
                    outputs = model(images)
                elif model_name == 'Early Fusion':
                    outputs = model(images, fm_maps)
                else:  # Late Fusion
                    outputs = model(images, vectors)
                
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2%} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2%}")
    
    return history

# -----------------------------
# Data Preparation
# -----------------------------
def find_original_image(base_path, category, base_filename):
    for ext in ['.png', '.jpg', '.jpeg', '.tif']:
        img_path = os.path.join(base_path, category, base_filename + ext)
        if os.path.exists(img_path):
            return img_path
    return None

all_files, all_labels = [], []

for category in CATEGORIES:
    feature_dir = os.path.join(FEATURES_PATH, category)
    if not os.path.isdir(feature_dir):
        continue
    for fname in os.listdir(feature_dir):
        if fname.endswith('.npy'):
            base_name = os.path.splitext(fname)[0]
            img_path = find_original_image(BASE_DATA_PATH, category, base_name)
            if img_path:
                all_files.append(img_path)
                all_labels.append(CLASS_TO_IDX[category])

train_files, val_files, train_labels, val_labels = train_test_split(
    all_files, all_labels, test_size=VALIDATION_SPLIT, random_state=42, stratify=all_labels
)

data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

train_dataset = FusionDatasetFM(train_files, train_labels, FEATURES_PATH, transform=data_transforms)
val_dataset   = FusionDatasetFM(val_files, val_labels, FEATURES_PATH, transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Total images: {len(all_files)} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# --- 6. Main Execution ---
# (No changes needed here)
models_to_run = {
    'Baseline': BaselineVGG16(num_classes=len(CATEGORIES)),
    'Early Fusion': EarlyFusionVGG16_FM(num_classes=len(CATEGORIES)),
    'Late Fusion': LateFusionVGG16_FM(num_classes=len(CATEGORIES))
}
all_histories = {}
for name, model in models_to_run.items():
    print(f"\n{'='*50}\nTraining Model: {name}\n{'='*50}")
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    history = train_and_validate(name, model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS)
    all_histories[name] = history
plt.style.use('seaborn-v0_8-whitegrid')

# First plot: Training and validation history over epochs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Model Training and Validation History (128x128)', fontsize=16)

# Plot validation accuracy
for name, history in all_histories.items():
    ax1.plot(history['val_acc'], 'o--', label=f'{name} Validation Acc')
ax1.set_title('Validation Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0, 1.0) # Set y-axis range from 0 to 1
ax1.legend()
ax1.grid(True)

# Plot validation loss
for name, history in all_histories.items():
    ax2.plot(history['val_loss'], 'x--', label=f'{name} Validation Loss')
ax2.set_title('Validation Loss over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Second plot: Final validation performance bar chart
final_val_acc = {name: h['val_acc'][-1] for name, h in all_histories.items()}
final_val_loss = {name: h['val_loss'][-1] for name, h in all_histories.items()}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Final Validation Performance Comparison (128x128)', fontsize=16)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Bar chart for final validation accuracy
bars1 = ax1.bar(final_val_acc.keys(), final_val_acc.values(), color=colors, edgecolor='black')
ax1.set_title('Final Validation Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0, 1.0) # Set y-axis range from 0 to 1
ax1.bar_label(bars1, fmt='{:.3%}')

# Bar chart for final validation loss
bars2 = ax2.bar(final_val_loss.keys(), final_val_loss.values(), color=colors, edgecolor='black')
ax2.set_title('Final Validation Loss')
ax2.set_ylabel('Loss')
ax2.bar_label(bars2, fmt='{:.4f}')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


import os
import numpy as np
import cv2
from tqdm.notebook import tqdm
import shutil

# --- Configuration ---
BASE_INPUT_PATH = '/kaggle/input/sarscov2-ctscan-dataset/' 
CATEGORIES = ['COVID', 'non-COVID']
# A new base path for the structured feature output
FEATURES_PATH = '/kaggle/working/features_up_to_10_10_structured/' 

# Set the maximum order for p and q
MAX_P = 10
MAX_Q = 10

# You can set a smaller number for testing, or None to process all images
SUBSET_SIZE_PER_CATEGORY = 500
IMG_WIDTH = 128
IMG_HEIGHT = 128

# --- Invariant Feature Calculation Functions (Unchanged) ---

def compute_invariant_features(image, max_p, max_q):
    """
    Compute the complete set of similarity invariant features If(p,q) for each channel.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be an RGB image (height, width, 3)")

    def compute_for_channel(channel):
        h, w = channel.shape
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        m00 = np.sum(channel)
        if m00 == 0: return np.zeros((max_p + 1, max_q + 1), dtype=complex)
        
        m10 = np.sum(x * channel)
        m01 = np.sum(y * channel)
        xc = m10 / m00
        yc = m01 / m00
        
        xs = x - xc
        ys = y - yc
        
        z = xs.astype(np.float64) + 1j * ys.astype(np.float64)
        zb = xs.astype(np.float64) - 1j * ys.astype(np.float64)
        
        c = np.zeros((max_p + 1, max_q + 1), dtype=complex)
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p >= q: # Optimization: only calculate for p >= q
                    c[p, q] = np.sum((z**p) * (zb**q) * channel)
                
        C_f = np.sqrt(c[0, 0].real) if c[0,0].real > 0 else 1.0
        H_f = np.angle(c[1, 0]) if np.abs(c[1, 0]) != 0 else 0.0
        
        I = np.zeros((max_p + 1, max_q + 1), dtype=complex)
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p >= q and C_f != 0:
                    I[p, q] = (C_f ** -(p + q + 2)) * np.exp(-1j * (p - q) * H_f) * c[p, q]
        return I

    channels = cv2.split(image)
    return [compute_for_channel(ch) for ch in channels]

def get_all_invariant_features(image, max_p, max_q):
    """
    Calculates all invariant features up to a specific (max_p, max_q) order.
    """
    resized_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    invariant_matrices = compute_invariant_features(resized_image, max_p, max_q)
    
    phases = np.zeros((max_p + 1, max_q + 1))
    amplitudes = np.zeros((max_p + 1, max_q + 1))

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p >= q:
                channel_phases = [np.angle(I[p, q]) for I in invariant_matrices]
                channel_amplitudes = [np.abs(I[p, q]) for I in invariant_matrices]
                
                phases[p, q] = np.mean(channel_phases)
                amplitudes[p, q] = np.mean(channel_amplitudes)
    
    return {'phases': phases, 'amplitudes': amplitudes}

# --- Main Execution ---
if __name__ == "__main__":
    if os.path.exists(FEATURES_PATH):
        print(f"Removing existing features directory: {FEATURES_PATH}")
        shutil.rmtree(FEATURES_PATH)
    
    print("Creating new directories for concatenated and flattened features...")
    for category in CATEGORIES:
        os.makedirs(os.path.join(FEATURES_PATH, 'channel_concatenated', category), exist_ok=True)
        os.makedirs(os.path.join(FEATURES_PATH, 'flattened_vector', category), exist_ok=True)

    # --- PASS 1: Calculate features and find min/max for normalization (Unchanged) ---
    print(f"\n--- Pass 1: Calculating features up to ({MAX_P},{MAX_Q}) and determining normalization stats ---")
    
    # Initialize stats with infinities to ensure first values are always set
    normalization_stats = {
        'phases': np.full((MAX_P + 1, MAX_Q + 1, 2), [np.inf, -np.inf]), # min, max
        'amplitudes': np.full((MAX_P + 1, MAX_Q + 1, 2), [np.inf, -np.inf]) # min, max
    }
    feature_cache = {}

    for category in CATEGORIES:
        print(f"\nProcessing category: {category}")
        category_path = os.path.join(BASE_INPUT_PATH, category)
        
        if not os.path.isdir(category_path):
            print(f"Warning: Directory not found for '{category}'. Skipping.")
            continue
            
        image_files = sorted(os.listdir(category_path))
        if SUBSET_SIZE_PER_CATEGORY is not None:
            image_files = image_files[:SUBSET_SIZE_PER_CATEGORY]

        for filename in tqdm(image_files, desc=f"Pass 1/2 - {category}"):
            image_path = os.path.join(category_path, filename)
            
            image = cv2.imread(image_path)
            if image is None: continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            features = get_all_invariant_features(image, max_p=MAX_P, max_q=MAX_Q)
            feature_cache[(category, filename)] = features

            for p in range(MAX_P + 1):
                for q in range(p + 1):
                    normalization_stats['phases'][p, q, 0] = min(normalization_stats['phases'][p, q, 0], features['phases'][p, q])
                    normalization_stats['phases'][p, q, 1] = max(normalization_stats['phases'][p, q, 1], features['phases'][p, q])
                    normalization_stats['amplitudes'][p, q, 0] = min(normalization_stats['amplitudes'][p, q, 0], features['amplitudes'][p, q])
                    normalization_stats['amplitudes'][p, q, 1] = max(normalization_stats['amplitudes'][p, q, 1], features['amplitudes'][p, q])

    print("\nGlobal Normalization Stats Calculation Complete.")

    # --- PASS 2: Normalize features and save in specified formats ---
    print("\n--- Pass 2: Normalizing and saving features as concatenated channels and flattened vectors ---")
    
    for (category, filename), features in tqdm(feature_cache.items(), desc="Pass 2/2 - Saving"):
        base_name = os.path.splitext(filename)[0]
        epsilon = 1e-9

        concatenated_feature_list = []
        flattened_feature_list = []

        # Iterate through all (p,q) orders to collect normalized features for this image
        for p in range(MAX_P + 1):
            for q in range(p + 1):
                min_phase, max_phase = normalization_stats['phases'][p, q]
                min_amplitude, max_amplitude = normalization_stats['amplitudes'][p, q]

                # Normalize phase and amplitude
                norm_phase = (features['phases'][p, q] - min_phase) / (max_phase - min_phase + epsilon)
                norm_amplitude = (features['amplitudes'][p, q] - min_amplitude) / (max_amplitude - min_amplitude + epsilon)
                
                # Append to the list for the flattened vector
                flattened_feature_list.append(norm_phase)
                flattened_feature_list.append(norm_amplitude)

                # Create constant maps and append to the list for channel concatenation
                concatenated_feature_list.append(np.full((IMG_HEIGHT, IMG_WIDTH), norm_phase, dtype=np.float32))
                concatenated_feature_list.append(np.full((IMG_HEIGHT, IMG_WIDTH), norm_amplitude, dtype=np.float32))

        # --- Save the Channel Concatenated version ---
        # Stack all feature maps along the last axis (channel)
        final_concatenated_array = np.stack(concatenated_feature_list, axis=-1)
        
        concat_output_path = os.path.join(FEATURES_PATH, 'channel_concatenated', category, f"{base_name}.npy")
        np.save(concat_output_path, final_concatenated_array)

        # --- Save the Flattened Vector version ---
        final_flattened_vector = np.array(flattened_feature_list, dtype=np.float32)

        flat_output_path = os.path.join(FEATURES_PATH, 'flattened_vector', category, f"{base_name}.npy")
        np.save(flat_output_path, final_flattened_vector)

    print(f"\nFeature pre-computation and saving is complete.")


import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# --- Configuration ---
BASE_IMAGE_PATH = '/kaggle/input/sarscov2-ctscan-dataset/'
FEATURES_PATH_FLAT = '/kaggle/working/features_up_to_10_10_structured/flattened_vector/'
CATEGORIES = ['COVID', 'non-COVID']
IMG_WIDTH = 128
IMG_HEIGHT = 128

# --- Data Split Configuration ---
VALIDATION_SPLIT_SIZE = 0.4 # Proportion of the total dataset to be used for validation

# Model Hyperparameters
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. Robust Data Preparation ---
print("Matching images to their pre-calculated feature files...")
image_files, flat_feature_files, labels = [], [], []
for i, category in enumerate(CATEGORIES):
    feature_dir = os.path.join(FEATURES_PATH_FLAT, category)
    if not os.path.isdir(feature_dir): continue
    for feature_filename in os.listdir(feature_dir):
        base_name = os.path.splitext(feature_filename)[0]
        image_filename = f"{base_name}.png"
        image_path = os.path.join(BASE_IMAGE_PATH, category, image_filename)
        if os.path.exists(image_path):
            image_files.append(image_path)
            flat_feature_files.append(os.path.join(FEATURES_PATH_FLAT, category, feature_filename))
            labels.append(i)
print(f"Found {len(image_files)} total valid image-feature pairs.")

# --- Simplified to a single train/validation split ---
X_train, X_val, y_train, y_val = train_test_split(
    list(zip(image_files, flat_feature_files)), 
    labels, 
    test_size=VALIDATION_SPLIT_SIZE, 
    random_state=42, 
    stratify=labels
)
print(f"Dataset split complete: {len(y_train)} training, {len(y_val)} validation.")
train_images, train_flat = zip(*X_train)
val_images, val_flat = zip(*X_val)


# --- 2. Custom PyTorch Dataset with Phase Encoding ---
class MultiModalDataset(Dataset):
    def __init__(self, image_paths, flat_paths, labels, mode='baseline', transform=None):
        self.image_paths, self.flat_paths = image_paths, flat_paths
        self.labels, self.mode, self.transform = labels, mode, transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        if self.transform: image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.mode == 'baseline':
            return image, label.unsqueeze(0)
        
        elif self.mode == 'vector_fusion':
            # --- NEW: Trigonometric Phase Encoding ---
            raw_features = np.load(self.flat_paths[idx])
            amplitudes = raw_features[0::2]
            phases = raw_features[1::2]
            
            # Encode phases as [cos(theta), sin(theta)]
            cos_phases = np.cos(phases)
            sin_phases = np.sin(phases)
            
            # Concatenate all features into a new, improved vector
            final_vector = np.concatenate([amplitudes, cos_phases, sin_phases])
            features = torch.from_numpy(final_vector).float()
            
            return image, features, label.unsqueeze(0)

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((IMG_HEIGHT, IMG_WIDTH)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create DataLoaders
train_ds_base = MultiModalDataset(train_images, train_flat, y_train, 'baseline', transform)
val_ds_base = MultiModalDataset(val_images, val_flat, y_val, 'baseline', transform)
train_loader_base = DataLoader(train_ds_base, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader_base = DataLoader(val_ds_base, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

train_ds_vector = MultiModalDataset(train_images, train_flat, y_train, 'vector_fusion', transform)
val_ds_vector = MultiModalDataset(val_images, val_flat, y_val, 'vector_fusion', transform)
train_loader_vector = DataLoader(train_ds_vector, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader_vector = DataLoader(val_ds_vector, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- 3. Dynamic Feature Size Calculation ---
if train_flat:
    temp_features = np.load(train_flat[0])
    NUM_AMPLITUDES = temp_features[0::2].shape[0]
    NUM_ENCODED_PHASES = temp_features[1::2].shape[0] * 2
    NUM_ENCODED_FEATURES = NUM_AMPLITUDES + NUM_ENCODED_PHASES
    print(f"Feature vector details: {NUM_AMPLITUDES} amplitudes + {NUM_ENCODED_PHASES} encoded phases = {NUM_ENCODED_FEATURES} total features.")
else:
    NUM_AMPLITUDES, NUM_ENCODED_PHASES, NUM_ENCODED_FEATURES = 0, 0, 0

# --- 4. Model Architectures ---
def build_baseline_model():
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)
    return model

class LateFusionVGG16(nn.Module):
    def __init__(self, num_features):
        super(LateFusionVGG16, self).__init__()
        vgg = models.vgg16(weights=None); self.features = vgg.features; self.avgpool = vgg.avgpool
        self.vector_mlp = nn.Sequential(nn.Linear(num_features, 128), nn.ReLU(), nn.Dropout(0.5))
        self.classifier = nn.Sequential(nn.Linear(25088 + 128, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 1))
    def forward(self, image, vector):
        img_out = torch.flatten(self.avgpool(self.features(image)), 1)
        vec_out = self.vector_mlp(vector)
        return self.classifier(torch.cat([img_out, vec_out], dim=1))

class IntermediateFusionVGG16(nn.Module):
    def __init__(self, num_features):
        super(IntermediateFusionVGG16, self).__init__()
        vgg = models.vgg16(weights=None); self.features = vgg.features; self.avgpool = vgg.avgpool
        self.feature_processor = nn.Sequential(nn.Linear(num_features, 128), nn.ReLU(), nn.Dropout(0.5))
        self.classifier = nn.Sequential(nn.Linear(25088 + 128, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 1))
    def forward(self, image, vector):
        img_out = torch.flatten(self.avgpool(self.features(image)), 1)
        vec_out = self.feature_processor(vector)
        return self.classifier(torch.cat([img_out, vec_out], dim=1))

# --- NEW: FiLM (Feature-wise Linear Modulation) Model ---
class FiLMLayer(nn.Module):
    def forward(self, feature_map, gamma, beta):
        return gamma.unsqueeze(-1).unsqueeze(-1) * feature_map + beta.unsqueeze(-1).unsqueeze(-1)

class FiLMFusionVGG16(nn.Module):
    def __init__(self, num_features, film_channels=512):
        super(FiLMFusionVGG16, self).__init__()
        vgg = models.vgg16(weights=None); self.features = vgg.features; self.avgpool = vgg.avgpool; self.classifier = vgg.classifier
        self.film_generator = nn.Sequential(nn.Linear(num_features, 512), nn.ReLU(), nn.Linear(512, film_channels * 2))
        self.film_layer = FiLMLayer()
        self.classifier[6] = nn.Linear(4096, 1)
    def forward(self, image, vector):
        feature_map = self.features(image)
        gamma, beta = torch.chunk(self.film_generator(vector), 2, dim=-1)
        modulated_map = self.film_layer(feature_map, gamma, beta)
        x = torch.flatten(self.avgpool(modulated_map), 1)
        return self.classifier(x)

# --- NEW: Amplitude-Gated Attention Model ---
class AmplitudeAttentionFusionVGG16(nn.Module):
    def __init__(self, num_amplitudes, num_encoded_phases):
        super(AmplitudeAttentionFusionVGG16, self).__init__()
        vgg = models.vgg16(weights=None); self.features = vgg.features; self.avgpool = vgg.avgpool
        self.phase_processor = nn.Sequential(nn.Linear(num_encoded_phases, 128), nn.ReLU())
        self.attention_generator = nn.Sequential(nn.Linear(num_amplitudes, 64), nn.Tanh(), nn.Linear(64, 128), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(25088 + 128, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 1))
    def forward(self, image, vector):
        amplitudes = vector[:, :NUM_AMPLITUDES]
        encoded_phases = vector[:, NUM_AMPLITUDES:]
        img_out = torch.flatten(self.avgpool(self.features(image)), 1)
        phase_features = self.phase_processor(encoded_phases)
        attention_gate = self.attention_generator(amplitudes)
        attended_features = phase_features * attention_gate
        combined = torch.cat([img_out, attended_features], dim=1)
        return self.classifier(combined)

# --- 5. Training and Validation Loop ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_type):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    if not len(train_loader.dataset) > 0: return model, history
    for epoch in range(num_epochs):
        model.train(); running_loss, correct_preds = 0.0, 0
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            is_vector = model_type == 'vector_fusion'
            images, vectors, labels = data if is_vector else (data[0], None, data[1])
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if is_vector: vectors = vectors.to(DEVICE)
            outputs = model(images, vectors) if is_vector else model(images)
            loss = criterion(outputs, labels); optimizer.zero_grad(); loss.backward(); optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct_preds += torch.sum((torch.sigmoid(outputs) > 0.5) == labels.data)
        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['train_acc'].append(correct_preds.double().item() / len(train_loader.dataset))
        model.eval(); running_loss, correct_preds = 0.0, 0
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                is_vector = model_type == 'vector_fusion'
                images, vectors, labels = data if is_vector else (data[0], None, data[1])
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                if is_vector: vectors = vectors.to(DEVICE)
                outputs = model(images, vectors) if is_vector else model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                correct_preds += torch.sum((torch.sigmoid(outputs) > 0.5) == labels.data)
        history['val_loss'].append(running_loss / len(val_loader.dataset))
        history['val_acc'].append(correct_preds.double().item() / len(val_loader.dataset))
        print(f"E{epoch+1}: Tr L: {history['train_loss'][-1]:.4f} A: {history['train_acc'][-1]:.4f} | Val L: {history['val_loss'][-1]:.4f} A: {history['val_acc'][-1]:.4f}")
    return model, history

# --- 6. Main Execution ---
criterion = nn.BCEWithLogitsLoss()
all_histories, all_val_accuracies = [], []

# --- Model Training Sequence ---
# Baseline Model
print("--- Training Baseline Model ---")
model_base = build_baseline_model().to(DEVICE)
optimizer_base = optim.Adam(model_base.parameters(), lr=LEARNING_RATE)
_, history_base = train_model(model_base, train_loader_base, val_loader_base, criterion, optimizer_base, EPOCHS, 'baseline')
all_histories.append(history_base); all_val_accuracies.append(max(history_base['val_acc'], default=0))
del model_base, optimizer_base; gc.collect(); torch.cuda.empty_cache()

# Late Fusion Model
print("\n--- Training Late Fusion Model ---")
model_late = LateFusionVGG16(NUM_ENCODED_FEATURES).to(DEVICE)
optimizer_late = optim.Adam(model_late.parameters(), lr=LEARNING_RATE)
_, history_late = train_model(model_late, train_loader_vector, val_loader_vector, criterion, optimizer_late, EPOCHS, 'vector_fusion')
all_histories.append(history_late); all_val_accuracies.append(max(history_late['val_acc'], default=0))
del model_late, optimizer_late; gc.collect(); torch.cuda.empty_cache()

# Intermediate Fusion Model
print("\n--- Training Intermediate Fusion Model ---")
model_inter = IntermediateFusionVGG16(NUM_ENCODED_FEATURES).to(DEVICE)
optimizer_inter = optim.Adam(model_inter.parameters(), lr=LEARNING_RATE)
_, history_inter = train_model(model_inter, train_loader_vector, val_loader_vector, criterion, optimizer_inter, EPOCHS, 'vector_fusion')
all_histories.append(history_inter); all_val_accuracies.append(max(history_inter['val_acc'], default=0))
del model_inter, optimizer_inter; gc.collect(); torch.cuda.empty_cache()

# FiLM Fusion Model
print("\n--- Training FiLM Fusion Model ---")
model_film = FiLMFusionVGG16(NUM_ENCODED_FEATURES).to(DEVICE)
optimizer_film = optim.Adam(model_film.parameters(), lr=LEARNING_RATE)
_, history_film = train_model(model_film, train_loader_vector, val_loader_vector, criterion, optimizer_film, EPOCHS, 'vector_fusion')
all_histories.append(history_film); all_val_accuracies.append(max(history_film['val_acc'], default=0))
del model_film, optimizer_film; gc.collect(); torch.cuda.empty_cache()

# Amplitude Attention Model
print("\n--- Training Amplitude-Gated Attention Model ---")
model_amp_att = AmplitudeAttentionFusionVGG16(NUM_AMPLITUDES, NUM_ENCODED_PHASES).to(DEVICE)
optimizer_amp_att = optim.Adam(model_amp_att.parameters(), lr=LEARNING_RATE)
_, history_amp_att = train_model(model_amp_att, train_loader_vector, val_loader_vector, criterion, optimizer_amp_att, EPOCHS, 'vector_fusion')
all_histories.append(history_amp_att); all_val_accuracies.append(max(history_amp_att['val_acc'], default=0))
del model_amp_att, optimizer_amp_att; gc.collect(); torch.cuda.empty_cache()

# --- 7. Final Visualization ---
model_names = ['Baseline', 'Late Fusion', 'Intermediate', 'FiLM Fusion', 'Amp. Attention']
print("\n--- Final Performance Report ---")
for name, val_acc in zip(model_names, all_val_accuracies):
    print(f"{name}: Best Validation Acc: {val_acc:.4f}")

# Plotting combined histories
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
for history, title in zip(all_histories, model_names):
    if history['val_acc']: ax1.plot(history['val_acc'], label=title)
ax1.set_title('Model Validation Accuracy Comparison', fontsize=14); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.legend()
for history, title in zip(all_histories, model_names):
    if history['val_loss']: ax2.plot(history['val_loss'], label=title)
ax2.set_title('Model Validation Loss Comparison', fontsize=14); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.legend()
plt.show()

# Plotting final bar chart
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=model_names, y=all_val_accuracies)
ax.set_title('Best Validation Accuracy Comparison', fontsize=16)
ax.set_ylabel('Best Validation Accuracy', fontsize=12); ax.set_xlabel('Model', fontsize=12)
ax.set_ylim(0.9, 1.0)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')
plt.xticks(rotation=15)
plt.show()