import os
import pandas as pd
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report,ConfusionMatrixDisplay,RocCurveDisplay,roc_auc_score,roc_curve, auc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader,WeightedRandomSampler
from torch.utils.data import Dataset as BaseDataset

def create_dataframe(anno_dir,normal_images_dir,test_size = 0.2):
    df_labels = pd.read_csv(anno_dir)
    df_labels['image'] = '../' + df_labels['image']
    
    # load normal images as new df
    rel_img_path = [os.path.relpath(x) for x in glob.glob(normal_images_dir)]
    df_nb = pd.DataFrame(rel_img_path,columns=['image'])
    df_nb['label'] = 0
    
    # concat dfs
    df = pd.concat([df_labels,df_nb], ignore_index=True)
    
    # encode labels
    df['label'] = df['label'].astype('category')
    df['label_encoded'] = df['label'].cat.codes.astype('int64')
    
    # create splits
    X_train, _ = train_test_split(df['image'].values,test_size=test_size,random_state=1,stratify=df['label'].values)
    df['split'] = ['train' if x in X_train else 'valid' for x in df['image'].values]
    df.rename(columns={'image':'file_path'},inplace=True)
    return df

def get_minority_transforms(img_size,img_mean,img_std):
    minority_transforms = A.Compose([
                        A.Resize(img_size, img_size),
                        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
                        A.Normalize(mean=img_mean, std=img_std),
                        ToTensorV2(),
                    ])
    return minority_transforms

def get_train_transforms(img_size,img_mean,img_std):
    train_transforms = A.Compose([
                        A.Resize(img_size, img_size),
                        A.OneOf(
                            [
                                A.CLAHE(p=1),
                                A.RandomBrightnessContrast(p=1),
                                A.RandomGamma(p=1),
                            ],
                            p=0.9,
                        ),
                        A.OneOf(
                            [
                                A.Sharpen(p=1),
                                A.Blur(blur_limit=3, p=1),
                                A.MotionBlur(blur_limit=3, p=1),
                            ],
                            p=0.9,
                        ),
                        A.OneOf(
                            [
                                A.RandomBrightnessContrast(p=1),
                                A.HueSaturationValue(p=1),
                            ],
                            p=0.9,
                        ),
                        A.Normalize(mean=img_mean, std=img_std),
                        ToTensorV2(),
                    ])
    return train_transforms

def get_valid_transforms(img_size,img_mean,img_std):
    valid_transforms = A.Compose([
                        A.Resize(img_size, img_size),
                        A.Normalize(mean=img_mean, std=img_std),
                        ToTensorV2(),
                    ])
    return valid_transforms

class Dataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations."""
    
    def __init__(
            self,
            df, 
            augmentation=None,
            minority_augmentation=None,
            visualize = False
    ):
        self.df = df.reset_index(drop=True)
        self.ids = self.df.loc[:,'file_path'].values
        
        self.augmentation = augmentation
        self.minority_augmentation = minority_augmentation
        self.visualize = visualize
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.ids[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
        label = self.df.loc[i,'label_encoded']
        
        # apply augmentations
        if self.minority_augmentation and label in [1,2,3,4,5]:
            image = self.minority_augmentation(image=image)['image']
        elif self.augmentation:
            image = self.augmentation(image=image)['image']
        
        # Revert Normalize to visualize the image
            if self.visualize:
                invTrans = A.Normalize(mean=[-x/y for x,y in zip(img_mean,img_std)],
                                       std=[1/x for x in img_std],
                                       max_pixel_value=1.0,
                                       always_apply=True)
                image = image.detach().cpu().numpy().transpose(1,2,0)
                image = invTrans(image=image)['image']
                image = (image*255).astype(np.uint8)
        
        return image, label
        
    def __len__(self):
        return len(self.ids)
    
def load_model(model_architecture,dropout,num_classes,dropout_percent=0.5,pretrained=True):
    model = timm.create_model(model_architecture, pretrained=pretrained, num_classes=num_classes)
    num_ftrs = model.get_classifier().in_features
    if dropout:
        model.classifier = nn.Sequential(
                                nn.Dropout(dropout_percent),
                                nn.Linear(num_ftrs,num_classes)
        )
    else:
        model.classifier = nn.Linear(num_ftrs, num_classes)

    return model

if __name__ == '__main__':
    
    # Configuration
    # Data
    anno_dir = '../../export/annotations/covidx_anno.csv'
    normal_images_dir = '../../data/Covidx/data_with_classes/*/normal/*'
    img_size = 224
    img_mean = IMAGENET_DEFAULT_MEAN
    img_std = IMAGENET_DEFAULT_STD
    df = create_dataframe(anno_dir,normal_images_dir)
    # Training and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 40
    model_architecture = 'convnext_small'
    num_classes = 6
    dropout = True
    k_folds = 5
    
    
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True,random_state=1)
    
    train_ids_dict = {}
    test_ids_dict = {}
    fold_train_acc = {}
    fold_val_acc = {}
    fold_train_loss = {}
    fold_val_loss = {}

    for fold, (train_ids,test_ids) in enumerate(kfold.split(np.zeros(len(df)),y=df['label_encoded'])):

        # Keep ids for later
        train_ids_dict[fold] = train_ids
        test_ids_dict[fold] = test_ids

        # Init data
        train_dataset = Dataset(
            df.loc[train_ids],
            augmentation=get_minority_transforms(img_size,img_mean,img_std),
        )

        valid_dataset = Dataset(
            df.loc[test_ids], 
            augmentation=get_valid_transforms(img_size,img_mean,img_std), 
        )
        
        class_weights = 1.0/df['label'].value_counts()
        sample_weights = [0] * len(train_dataset)
        
        for idx, (data,label) in enumerate(train_dataset):
            sample_weights[idx] = class_weights[label]
            
        sampler = WeightedRandomSampler(sample_weights,num_samples=len(sample_weights),replacement=True)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=12,sampler=sampler)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

        # Init model
        model = load_model(model_architecture,dropout,num_classes)
        model = model.to(device)

        # Init optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),lr=0.0001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs-1)
        scaler = torch.cuda.amp.GradScaler()

        CHECKPOINT = f'./models/{model_architecture}_covidx_severity_oversampling_weak_fold{fold}.pth'

        print(f'FOLD {fold}')
        print('--------------------------------')

        # Training loop
        train_acc_list = []
        val_acc_list = []
        train_loss_list = []
        val_loss_list = []
        val_loss_min = np.Inf

        for epoch in range(epochs):
            model.train()
            train_loss = []
            train_running_corrects = 0
            val_running_corrects = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    train_loss.append(loss.item())

                    _, predicted = torch.max(outputs.data, 1)
                    #_,labels = torch.max(labels.data, 1)
                    train_running_corrects += torch.sum(predicted == labels.data)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()

                #loop.set_description('Epoch {:02d}/{:02d} | LR: {:.5f}'.format(epoch, epochs-1, optimizer.param_groups[0]['lr']))
                #loop.set_postfix(loss=np.mean(train_loss))

            train_loss = np.mean(train_loss)
            train_epoch_acc = train_running_corrects.double() / len(train_loader.dataset)

            model.eval()

            val_loss = 0

            # Validation loop
            with torch.cuda.amp.autocast(), torch.no_grad():    
                for images, labels in valid_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    #_,labels = torch.max(labels.data, 1)
                    val_running_corrects += torch.sum(predicted == labels.data)

            val_loss /= len(valid_loader.dataset)
            val_epoch_acc = val_running_corrects.double() / len(valid_loader.dataset)

            print(f'Epoch {epoch}: train loss: {train_loss:.5f} | train acc: {train_epoch_acc:.3f} | val_loss: {val_loss:.5f} | val acc: {val_epoch_acc:.3f}')

            train_acc_list.append(train_epoch_acc.item())
            val_acc_list.append(val_epoch_acc.item())
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            if val_loss < val_loss_min:
                    print(f'Valid loss improved from {val_loss_min:.5f} to {val_loss:.5f} saving model to {CHECKPOINT}')
                    val_loss_min = val_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), CHECKPOINT)

            print(f'Best epoch {best_epoch} | val loss min: {val_loss_min:.5f}')

        fold_train_acc[fold] = train_acc_list
        fold_val_acc[fold] = val_acc_list
        fold_train_loss[fold] = train_loss_list
        fold_val_loss[fold] = val_loss_list

        # Delete model just to be sure
        del loss, model, optimizer
        torch.cuda.empty_cache()
    
    # Final Results Output
    avg = 0.0
    for fold,val in fold_val_acc.items():
        print(f'Highest val_acc for fold {fold}: {np.max(val):.3f}')
        avg += np.max(val)
    print(f'Average for all folds: {avg/len(fold_val_acc.items()):.3f}')
    
    #Saving Metrics
    df = pd.DataFrame()
    for metric,name in zip([fold_train_acc,fold_train_loss,fold_val_acc,fold_val_loss],['train_acc','train_loss','val_acc','val_loss']):
        dffold = pd.DataFrame.from_dict(fold_val_acc,orient='columns')
        dffold.columns = [f'fold{x}_{name}' for x in range(len(fold_val_acc))]
        dffold = dffold.rename_axis('epochs')
        df = pd.concat([df,dffold],axis=1)
    df.to_csv(f'./logs/{model_architecture}_covidx_severity_oversampling_weak_cvmetrics.csv')
    #pd.DataFrame.from_dict([train_ids_dict,test_ids_dict],orient='columns').to_csv('./logs/oversampling_cv_splits.csv')
