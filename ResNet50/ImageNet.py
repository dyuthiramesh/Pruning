import os
import deeplake
import torch as th
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Initialize random seed for reproducibility
seed = 1787
th.manual_seed(seed)
th.cuda.manual_seed(seed)
th.cuda.manual_seed_all(seed)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False

# Set device
device = th.device("cuda" if th.cuda.is_available() else "cpu")

class DeepLakeImageNet(Dataset):
    def __init__(self, deeplake_dataset, transform=None):
        self.dataset = deeplake_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        image, label = data['images'], data['labels']
        image = image.numpy().transpose((1, 2, 0))  # Convert to HWC format
        if self.transform:
            image = self.transform(image)
        return image, label.numpy()

class Data:
    def __init__(self, gpu, data_dir, batch_size):
        pin_memory = False
        if gpu is not None:
            pin_memory = True

        scale_size = 224

        # Set up Deep Lake local storage paths
        train_path = os.path.join(data_dir, 'imagenet-train')
        val_path = os.path.join(data_dir, 'imagenet-val')

        # Load ImageNet training dataset
        if not os.path.exists(train_path):
            train_ds = deeplake.load('hub://activeloop/imagenet-train', dest=train_path)
        else:
            train_ds = deeplake.load(train_path)

        # Load ImageNet validation dataset
        if not os.path.exists(val_path):
            val_ds = deeplake.load('hub://activeloop/imagenet-val', dest=val_path)
        else:
            val_ds = deeplake.load(val_path)

        # Normalize transformation
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Training transformations
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(scale_size),
            transforms.ToTensor(),
            normalize,
        ])

        # Validation transformations
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(scale_size),
            transforms.ToTensor(),
            normalize,
        ])

        # Create Deep Lake datasets
        train_dataset = DeepLakeImageNet(train_ds, transform=train_transform)
        val_dataset = DeepLakeImageNet(val_ds, transform=val_transform)

        # Create PyTorch data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=pin_memory)

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=pin_memory)

# Usage
data_dir = '../data'  # Directory to save ImageNet dataset
batch_size = 32
gpu = 0 if th.cuda.is_available() else None

data = Data(gpu=gpu, data_dir=data_dir, batch_size=batch_size)

# Example usage of data loaders
for images, labels in data.train_loader:
    print(images.shape, labels.shape)
    break
