import os
import pickle
from PIL import Image

from torch.utils.data import Dataset


DEFAULT_PATH = {'mdfa_test': 'E:\Infrared Small Target Dataset\MDvsFA_test',
                'mdfa_train': 'E:\Infrared Small Target Dataset\MDvsFA_train',
                'sirst': 'E:\Infrared Small Target Dataset\Sirst_test'}
'''
DEFAULT_PATH = {'mdfa_test': '../data/Infrared Small Target Dataset/MDvsFA_test',
                'mdfa_train': '../data/Infrared Small Target Dataset/MDvsFA_train',
                'sirst': '../data/Infrared Small Target Dataset/Sirst_test'}
'''



class MDFADataset(Dataset):
    def __init__(self, root, split, transform=None, target_transform=None):
        super(MDFADataset, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        if split == 'train':
            self._parser_train_folder()
        else:
            self._parser_test_folder()

    def _parser_train_folder(self):
        if os.path.exists('./img_file_list.pkl'):
            with open('./img_file_list.pkl', 'rb') as f:
                save_cache = pickle.load(f)
                self.x_files = save_cache["train_x"]
                self.y_files = save_cache["train_y"]

        else:
            files_all = os.listdir(self.root)
            self.x_files = [file for file in files_all if file[7] == '1']
            self.y_files = [file for file in files_all if file[7] == '2']
            self.x_files.sort(key=lambda x:int(x.split('_')[0]))
            self.y_files.sort(key=lambda x:int(x.split('_')[0]))

    def _parser_test_folder(self):
        x_files = os.listdir(os.path.join(self.root, 'test_org'))
        y_files = os.listdir(os.path.join(self.root, 'test_gt'))

        self.x_files = [os.path.join('test_org', filename) for filename in x_files]
        self.y_files = [os.path.join('test_gt', filename) for filename in y_files]
        self.x_files.sort(key=lambda x:int(os.path.split(x)[1].split('.')[0]))
        self.y_files.sort(key=lambda x:int(os.path.split(x)[1].split('.')[0]))

    def _getitem_(self, index):
        x_filename = self.x_files[index]
        y_filename = self.y_files[index]

        x = Image.open(os.path.join(self.root, x_filename))
        y = Image.open(os.path.join(self.root, y_filename))

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def __getitem__(self, index):
        return self._getitem_(index)

    def __len__(self):
        return len(self.x_files)


class SirstDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(SirstDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self._parser_folder()

    def _parser_folder(self):
        x_files = os.listdir(os.path.join(self.root, 'images'))
        y_files = os.listdir(os.path.join(self.root, 'gts'))

        self.x_files = [os.path.join('images', filename) for filename in x_files]
        self.y_files = [os.path.join('gts', filename) for filename in y_files]
        self.x_files.sort(key=lambda x:int(x.split('_')[1].split('.')[0]))
        self.y_files.sort(key=lambda x:int(x.split('_')[1]))

    def _getitem_(self, index):
        x_filename = self.x_files[index]
        y_filename = self.y_files[index]

        x = Image.open(os.path.join(self.root, x_filename))
        y = Image.open(os.path.join(self.root, y_filename))

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def __getitem__(self, index):
        return self._getitem_(index)

    def __len__(self):
        return len(self.x_files)