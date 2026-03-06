from torchvision import transforms

train_transform = transforms.Compose([

    transforms.Resize((128,128)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485],
        std=[0.229]
    )
])

val_test_transform = transforms.Compose([

    transforms.Resize((128,128)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485],
        std=[0.229]
    )
])