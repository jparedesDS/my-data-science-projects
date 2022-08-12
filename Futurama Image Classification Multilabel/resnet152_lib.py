import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import Torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms, models
import torchvision.models as models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class FuturamaDataset(Dataset):
    def __init__(self, data_path, train_images_path, transform=None):
        self.data_path = data_path
        self.train_images_path = train_images_path
        self.data_frame = pd.read_csv(data_path)
        self.transform = transform
        self.n_samples = len(self.data_frame.index)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img_path = os.path.join(self.train_images_path, self.data_frame["file"][idx])

        image = self.transform(Image.open(img_path).convert("RGB"))

        isLeela = self.data_frame["isLeela"][idx]
        isFry = self.data_frame["isFry"][idx]
        isBender = self.data_frame["isBender"][idx]

        sample = {
            "image": image,
            "labels": {"isLeela": isLeela, "isFry": isFry, "isBender": isBender},
        }
        return sample


class FuturamaResnet(nn.Module):
    def __init__(self, resnet_model, hidden_mlp, drop_out):
        super().__init__()

        def mlp(layer_in, hidden, layer_out):
            return (
                nn.Dropout(p=drop_out),
                nn.Linear(layer_in, hidden),
                nn.Tanh(),
                nn.Linear(hidden, layer_out),
            )

        if resnet_model == "resnet152":
            print("using resnet152")

            self.resnet = models.resnet152(
                weights=torchvision.models.ResNet152_Weights.DEFAULT
            )

            self.model_wo_fc = nn.Sequential(*(list(self.resnet.children())[:-1]))

            self.isLeela = nn.Sequential(
                nn.Dropout(p=drop_out), *mlp(2048, hidden_mlp, 2)
            )
            self.isFry = nn.Sequential(
                nn.Dropout(p=drop_out), *mlp(2048, hidden_mlp, 2)
            )
            self.isBender = nn.Sequential(
                nn.Dropout(p=drop_out), *mlp(2048, hidden_mlp, 2)
            )
        elif resnet_model == "resnet34":
            print("using resnet34")

            self.resnet = models.resnet34(
                weights=torchvision.models.ResNet34_Weights.DEFAULT
            )

            self.model_wo_fc = nn.Sequential(*(list(self.resnet.children())[:-1]))

            self.isLeela = nn.Sequential(
                nn.Dropout(p=drop_out), *mlp(512, hidden_mlp, 2)
            )
            self.isFry = nn.Sequential(nn.Dropout(p=drop_out), *mlp(512, hidden_mlp, 2))

            self.isBender = nn.Sequential(
                nn.Dropout(p=drop_out), *mlp(512, hidden_mlp, 2)
            )

    def forward(self, x):
        x = self.model_wo_fc(x)
        x = torch.flatten(x, 1)

        return {
            "isLeela": self.isLeela(x),
            "isFry": self.isFry(x),
            "isBender": self.isBender(x),
        }


def calc_results(outputs, labels):

    with torch.no_grad():

        predicteds = []
        error_class = 0

        for key in outputs.keys():
            _, predicted = torch.max(outputs[key], 1)
            predicteds.append(predicted)

        for label_class, predicted_class in zip(labels, predicteds):
            error_class += torch.sum(label_class != predicted_class)

    return error_class


def criterion(loss_func, outputs, pictures):
    losses = 0

    for key in outputs.keys():

        losses += loss_func(outputs[key], pictures["labels"][key].to(device))

    return losses


def training(
    model, device, writer, lr_rate, lr_decay, epochs, train_loader, val_loader
):
    num_epochs = epochs

    epoch_losses_train = []
    epoch_losses_test = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, lr_decay, verbose=True
    )
    n_total_steps_train = len(train_loader)
    n_total_steps_test = len(val_loader)

    loss_func = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):

        model.train()
        losses = []

        for i, pictures in enumerate(train_loader):
            images = pictures["image"].to(device)
            pictures = pictures

            outputs = model(images)

            loss = criterion(loss_func, outputs, pictures)

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        checkpoint_loss = torch.tensor(losses).mean().item()
        epoch_losses_train.append(checkpoint_loss)
        if writer is not None:
            writer.add_scalar("Loss/train", checkpoint_loss, epoch)

        print(
            f"Epoch Train[{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps_train}], Loss: {checkpoint_loss:.4f}"
        )

        if len(val_loader.dataset) > 0:
            model.eval()
            losses = []
            errors_class = 0.0
            for i, pictures in enumerate(val_loader):

                images = pictures["image"].to(device)
                pictures = pictures

                outputs = model(images)

                loss = criterion(loss_func, outputs, pictures)

                losses.append(loss.item())

                labels = [
                    pictures["labels"][picture].to(device)
                    for picture in pictures["labels"]
                ]

                errors_class += calc_results(outputs, labels)

            checkpoint_loss = torch.tensor(losses).mean().item()
            epoch_losses_test.append(checkpoint_loss)

            print(
                f"Epoch Test [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps_test}], Loss: {checkpoint_loss:.4f}"
            )

            class_failed = errors_class / (3 * len(val_loader.dataset))

            if writer is not None:
                writer.add_scalar("Loss/test", checkpoint_loss, epoch)

                writer.add_scalar("Accuracy_class/test", class_failed, epoch)

            print(f"Percentage of classes failed: {class_failed}")

        exp_scheduler.step()


def create_submission(
    model, sample_sub_path, test_img_path, dst_path, data_mean, data_std
):

    sample_submission = pd.read_csv(sample_sub_path)

    test_img_name_list = sample_submission["file"].to_list()

    # create an empty answer dataframe
    submission_df = pd.DataFrame(
        {
            "file": test_img_name_list,
            "isLeela": np.zeros(1000, dtype=int),
            "isFry": np.zeros(1000, dtype=int),
            "isBender": np.zeros(1000, dtype=int),
        }
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std),
        ]
    )

    model.eval()
    for n, row in enumerate(submission_df.iterrows()):

        test_img = test_transform(
            Image.open(os.path.join(test_img_path, row[1][0])).convert("RGB")
        )

        test_img = test_img.to(device)
        test_img = test_img[None, :, :, :]

        outputs = model(test_img)

        submission_df.at[n, "isLeela"] = torch.max(outputs["isLeela"], 1)[1].item()
        submission_df.at[n, "isFry"] = torch.max(outputs["isFry"], 1)[1].item()
        submission_df.at[n, "isBender"] = torch.max(outputs["isBender"], 1)[1].item()

    submission_df.to_csv(dst_path, index=False)