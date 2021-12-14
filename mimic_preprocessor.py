import pickle
import os
from dicom_parser import Image
from tqdm import tqdm

def get_patient_position(path):
    dcm = Image(path)
    patient_position = dcm.header.get('ViewCodeSequence')[0].get('CodeMeaning')
    return patient_position


def read_free_text(patient_path, free_text):
    """
    Usecase
    # read_free_text('/home/nas1_userE/gyuhyeonsim/physionet.org/files/mimic-cxr/2.0.0/files/p16/p16802198', 's54334314.txt')
    """
    f = open(patient_path+'/'+free_text, 'r')
    strings = f.read()
    f.close()
    return strings

def get_valid_patients(file_list):
    """
    Variables for META INFORMATION is capitalized.
    """

    count = 0
    error_cases = list()
    valid_cases = list()
    error_num = 0
    valid_num = 0
    study_list = []

    for i in tqdm(file_list):
        file_list = ft_path_dict[i]
        patient_id = i.split('/')[-1]

        for j in file_list:
            # FINDINGS가 없으면 제외(txt field)
            free_text = read_free_text(i, j)
            if 'FINDINGS' not in free_text:
                error_cases.append(directory_path)
                print('no findings: {}'.format(j))
                continue

            file_id = j.split('.')[0]
            directory_path = i + '/' + file_id

            # dcm list
            dcm_list = os.listdir(directory_path)
            dcm_list.remove('index.html')

            # One patient's study can have many dicom files
            for k in dcm_list:
                print('candidate: {}'.format(k))
                try:
                    PATIENT_POSITION = get_patient_position(directory_path + '/' + k)
                except:
                    print('ERROR: {}'.format(k))
                    error_num += 1
                    continue

                if 'lateral' not in PATIENT_POSITION:
                    study_list.append(directory_path + '/' + k)
                    valid_num += 1
                else:
                    error_num += 1
                    print(k, PATIENT_POSITION)

            if len(dcm_list) == 0:
                error_cases.append(directory_path)
                continue

        valid_cases.append(directory_path)
    print('VALID NUM:{}, ERROR NUM: {}'.format(valid_num, error_num))
    return error_cases, valid_cases, study_list

PATH = '/home/nas1_userE/gyuhyeonsim/'
with open(PATH+'train_patient.pickle', 'rb') as f:
    train_data = pickle.load(f)['file_list']
with open(PATH+'valid_patient.pickle', 'rb') as f:
    valid_data = pickle.load(f)['file_list']
with open(PATH+'test_patient.pickle', 'rb') as f:
    test_data = pickle.load(f)['file_list']
with open(PATH+'patient2study.pickle', 'rb') as f:
    ft_path_dict = pickle.load(f)

from tqdm import tqdm
train_ec, train_vc, train_study_list = get_valid_patients(train_data)
valid_ec, valid_vc, valid_study_list = get_valid_patients(valid_data)
test_ec, test_vc, test_study_list = get_valid_patients(test_data)


import pickle
with open('train_file_list.pickle', 'wb') as f:
    pickle.dump({'file_list': train_study_list}, f, pickle.HIGHEST_PROTOCOL)
with open('valid_file_list.pickle', 'wb') as f:
    pickle.dump({'file_list': valid_study_list}, f, pickle.HIGHEST_PROTOCOL)
with open('test_file_list.pickle', 'wb') as f:
    pickle.dump({'file_list': test_study_list}, f, pickle.HIGHEST_PROTOCOL)