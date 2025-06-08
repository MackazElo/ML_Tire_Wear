import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32, image_size=(224, 224), augment=False):
    """
    Przygotowuje DataLoadery dla zbiorów treningowego i walidacyjnego/testowego.
    
    Args:
        data_dir (str): Ścieżka do katalogu z podfolderami 'train', 'val', 'test'.
        batch_size (int): Liczba próbek w jednej partii.
        image_size (tuple): Rozmiar obrazów do skalowania.
        augment (bool): Czy zastosować augmentację danych w treningu.

    Returns:
        dict: Słownik z DataLoaderami dla 'train', 'val', 'test'.
    """

    base_transforms = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        *base_transforms
    ]) if augment else transforms.Compose(base_transforms)

    test_transforms = transforms.Compose(base_transforms)

    loaders = {}
    for split in ['train', 'val', 'test']:
        path = os.path.join(data_dir, split)
        if not os.path.exists(path):
            continue  
        transform = train_transforms if split == 'train' else test_transforms

        dataset = datasets.ImageFolder(root=path, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
        loaders[split] = loader
    print("Dostępne foldery:", os.listdir(data_dir))
    for split in ['train', 'val', 'test']:
        path = os.path.join(data_dir, split)
        print(f"Sprawdzam: {path} | Istnieje: {os.path.exists(path)}")

    return loaders
