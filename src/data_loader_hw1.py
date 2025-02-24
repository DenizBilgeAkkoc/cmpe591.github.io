import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RobotDataset(Dataset):
    def __init__(self, before_images, actions, after_images, transform=None):
        self.before_images = before_images
        self.actions = actions
        self.after_images = after_images
        self.transform = transform

    def __len__(self):
        return len(self.before_images)

    def __getitem__(self, idx):
        before_img = self.before_images[idx]
        action = self.actions[idx]
        after_img = self.after_images[idx]

        # Convert action to one-hot
        action_onehot = torch.nn.functional.one_hot(action.to(torch.int64), num_classes=4).float()

        if self.transform:
            before_img = self.transform(before_img)
            after_img = self.transform(after_img)

        return before_img, action_onehot, after_img
    


def get_data_loaders(batch_size=16, num_workers=0):
    # Load data
    img_tensor = (torch.tensor(np.load("before-image-1000.npy").astype(np.float32), dtype=torch.float32))/255.0
    act_tensor = torch.tensor(np.load("actions-1000.npy"), dtype=torch.float32)
    after_img_tensor = (torch.tensor(np.load("after-image-1000.npy").astype(np.float32), dtype=torch.float32))/255.0

    def rgb_to_grayscale_luminance(image):
        grayscale_image = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        return grayscale_image.unsqueeze(1)

    # Convert to grayscale
    img = rgb_to_grayscale_luminance(img_tensor)
    after_img = rgb_to_grayscale_luminance(after_img_tensor)

    # Split into train and test sets
    train_size = 800

    # Create datasets
    train_dataset = RobotDataset(
        before_images=img[:train_size],
        actions=act_tensor[:train_size],
        after_images=after_img[:train_size]
    )

    test_dataset = RobotDataset(
        before_images=img[train_size:],
        actions=act_tensor[train_size:],
        after_images=after_img[train_size:]
    )

    # Create data loaders with modified settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # Set to 0 for debugging
        pin_memory=True,
        persistent_workers=False,  # Disable persistent workers
        drop_last=True  # Drop incomplete last batch
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,  # Set to 0 for debugging
        pin_memory=True,
        persistent_workers=False,  # Disable persistent workers
        drop_last=True  # Drop incomplete last batch
    )

    return train_loader, test_loader