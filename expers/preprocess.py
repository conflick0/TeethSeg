import os
import shutil
from pathlib import PurePath

from monai.transforms import (
    LoadImage
)

from data_utils.data_loader_utils import split_data_dicts
from data_utils.teeth_dataset import get_data_dicts
from data_utils.io import save_json


def get_rel_pth(pth):
    return os.path.join(*PurePath(pth).parts[-2:])


def get_rel_data_dicts(data_dicts):
    out_data_dicts = []
    for data_dict in data_dicts:
        out_data_dict = {}
        for k in data_dict.keys():
            out_data_dict[k] = get_rel_pth(data_dict[k])
        out_data_dicts.append(out_data_dict)
    return out_data_dicts


def build_data_dicts(
        get_data_dicts_fn,
        src_data_dir,
        dst_data_json,
        split_train_ratio,
        num_fold,
        fold
    ):

    data_dicts = get_data_dicts_fn(src_data_dir)
    train_files, val_files, test_files = split_data_dicts(
        data_dicts,
        fold,
        split_train_ratio,
        num_fold
    )
    out_data_dicts = {
        'train': get_rel_data_dicts(train_files),
        'val': get_rel_data_dicts(val_files),
        'test': get_rel_data_dicts(test_files)
    }
    save_json(out_data_dicts, dst_data_json)



def extract_data_by_lbl_num(target_lbl_num, src_data_dir, dst_data_dir):
    '''extract same lbl length data'''
    os.makedirs(os.path.join(dst_data_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(dst_data_dir, 'label'), exist_ok=True)

    loader = LoadImage()

    data_dicts = get_data_dicts(src_data_dir)

    for data_dict in data_dicts:
        # read lbl cls
        lbl = loader(data_dict['label'])[0]
        lbl_cls = lbl.flatten().unique()
        # save the label num of match
        if len(lbl_cls) == target_lbl_num:
            print(data_dict)
            dst_img_pth = os.path.join(dst_data_dir, get_rel_pth(data_dict['image']))
            dst_lbl_pth = os.path.join(dst_data_dir, get_rel_pth(data_dict['label']))

            # copy file
            shutil.copyfile(data_dict['image'], dst_img_pth)
            shutil.copyfile(data_dict['label'], dst_lbl_pth)


def extract_data_by_file_name(file_names, src_data_dir, dst_data_dir):
    '''extract data by file names'''
    os.makedirs(os.path.join(dst_data_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(dst_data_dir, 'label'), exist_ok=True)

    data_dicts = get_data_dicts(src_data_dir)
    for data_dict in data_dicts:
        file_name = PurePath(data_dict['image']).parts[-1].split('.')[0]
        if file_name in file_names:
            print(file_name)
            dst_img_pth = os.path.join(dst_data_dir, get_rel_pth(data_dict['image']))
            dst_lbl_pth = os.path.join(dst_data_dir, get_rel_pth(data_dict['label']))

            # copy file
            shutil.copyfile(data_dict['image'], dst_img_pth)
            shutil.copyfile(data_dict['label'], dst_lbl_pth)


if __name__ == '__main__':
    # extract same lbl num
    # target_lbl_num = 29
    # src_data_dir = r'D:\home\school\ntut\dataset\teeth\Train'
    # dst_data_dir = r'D:\home\school\ntut\dataset\teeth\data_pp'
    # extract_data_by_lbl_num(target_lbl_num, src_data_dir, dst_data_dir)

    # extract by file name
    # src_data_dir = r'D:\home\school\ntut\dataset\teeth\data_pp'
    # dst_data_dir = r'D:\home\school\ntut\dataset\teeth\data'
    # file_names = [
    #     '1001275319_20180114',
    #     '1001382496_20180423',
    #     '1001382496_20201206',
    #     '1001470164_20180114',
    #     '1001487462_20180109',
    #     '1001487462_20180527',
    #     '1001487462_20190427',
    #     '1001162439_20150708'
    # ]
    # extract_data_by_file_name(file_names, src_data_dir, dst_data_dir)

    # build data dict
    data_dir = r'/nfs/Workspace/dataset/teeth/cindydata'
    dst_data_json = os.path.join(data_dir, 'data.json')

    build_data_dicts(
        get_data_dicts,
        src_data_dir=data_dir,
        dst_data_json=dst_data_json,
        split_train_ratio=0.8,
        num_fold=2,
        fold=1,
    )
