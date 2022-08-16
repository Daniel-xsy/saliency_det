import os 
import pickle
import torchvision.transforms as tfs

from dataset import DEFAULT_PATH, MDFADataset

if __name__=='__main__':

    transforms = tfs.Compose([
        tfs.Resize((224, 224)),
        tfs.CenterCrop((224, 224)),
        tfs.Grayscale(num_output_channels=1),
        tfs.ToTensor(),
        tfs.Normalize(mean=0.5, std=0.5)
    ])

    target_transforms = tfs.Compose([
        tfs.Resize((224, 224)),
        tfs.CenterCrop((224, 224)),
        tfs.Grayscale(num_output_channels=1),
        tfs.ToTensor()
    ])

    root = DEFAULT_PATH['mdfa_train']
    train_dataset = MDFADataset(root=root, split='train', transform=transforms, target_transform=target_transforms)
    train_x_file_list = []
    train_y_file_list = []
    for i in range(len(train_dataset)):
        print('processing %i / %i' % (i, len(train_dataset)), end='\r')
        try:
            x, y = train_dataset[i]
            train_x_file_list.append(train_dataset.x_files[i])
            train_y_file_list.append(train_dataset.x_files[i])
        except:
            pass

    with open('img_file_list.pkl', 'wb') as f:
        pickle.dump(
            {
                "train_x": train_x_file_list,
                "train_y": train_y_file_list
            }, f)
