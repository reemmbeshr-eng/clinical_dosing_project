from torchvision import transforms

train_transform = transforms.Compose([

    transforms.Resize((128,128)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(5),

    transforms.ToTensor(),

    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

])

val_test_transform = transforms.Compose([

    transforms.Resize((128,128)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485],
        std=[0.229]
    )
])