import SimpleITK as sitk
import pickle
import torch
import numpy as np
import torchvision.transforms as transforms


class MIMIC_CXR(torch.utils.data.dataset.Dataset):
    def __init__(self, split):

        ## 파일은 완료되면 nas로 이동 예정
        if split == 'train':
            with open('/home/gyuhyeonsim/asan/train_file_list.pickle', 'rb') as f:
                self.file_list = pickle.load(f)['file_list']
                self.example = '/home/nas1_userE/gyuhyeonsim/physionet.org/files/mimic-cxr/2.0.0/files/p11/p11052737/s58244732/a63dd2df-4ff38ca9-db5e3c78-3ce15db6-5ef651cd.dcm'

        elif split == 'valid':
            with open('/home/gyuhyeonsim/asan/valid_file_list.pickle', 'rb') as f:
                self.file_list = pickle.load(f)['file_list']

        elif split == 'test':
            with open('/home/gyuhyeonsim/asan/test_file_list.pickle', 'rb') as f:
                self.file_list = pickle.load(f)['file_list']
        self.resize = transforms.Resize((256, 256))

    def __getitem__(self, idx):
        # target_file = self.file_list[idx]
        target_file = self.example
        file_meta_info = self.example.split('/')
        base_path = "/".join(file_meta_info[:-3])
        patient_id = file_meta_info[-3]
        study_id = file_meta_info[-2]
        file_id = file_meta_info[-1]

        # Get report
        free_text_file = self.read_free_text(base_path + '/' + patient_id, study_id + '.txt')
        """
        Report에서 FINDINGS 찾는 코드만 부탁드려요.
        """

        # Get image
        dcm_image = sitk.ReadImage(target_file)
        dcm_np = sitk.GetArrayFromImage(dcm_image)
        dcm_tensor = torch.from_numpy(dcm_np.astype(np.float))
        dcm_tensor = dcm_tensor.float()

        # Preprocessing
        dcm_tensor = self.resize(dcm_tensor)

        # noramlize

        return {'report': free_text_file,
                'image': dcm_tensor}

    def read_free_text(self, patient_path, free_text):
        """
        Read free text file
        """
        f = open(patient_path + '/' + free_text, 'r')
        strings = f.read()
        f.close()
        return strings

    def noralize_xray(self, np):
        # mean: 2549.440547823737
        # std: 1490.4394489321774
        np = (np-2549.440547823737)/1490.4394489321774
        np = (np-np.min())/(np.max()-np.min())
        return np

# read_free_text('/home/nas1_userE/gyuhyeonsim/physionet.org/files/mimic-cxr/2.0.0/files/p16/p16802198', 's54334314.txt')