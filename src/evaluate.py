import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=2, pretrained=True):
    """
    Zwraca model ResNet18 dostosowany do liczby klas.

    Args:
        num_classes (int): Liczba klas do rozpoznania.
        pretrained (bool): Czy załadować wagi wstępnie wytrenowane na ImageNet.

    Returns:
        nn.Module: Model sieci neuronowej.
    """
    model = models.resnet18(pretrained=pretrained)

    # Zamieniamy ostatnią warstwę FC na dostosowaną do liczby klas
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
