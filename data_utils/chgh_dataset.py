import os

from data_utils.data_loader import MyDataLoader, get_dl
from transforms.teeth_transform import get_train_transform, get_val_transform, get_inf_transform


def get_data_dicts(data_dir):
    patient_dirs = sorted(os.listdir(data_dir), key=lambda x: int(x.split('_')[-1]))
    data_dicts = []
    for patient_dir in patient_dirs:
        data_dicts.append({
            "image": os.path.join(os.path.join(data_dir, patient_dir, f'{patient_dir}.nii.gz')),
            "label": os.path.join(os.path.join(data_dir, patient_dir, f'{patient_dir}_gt.nii.gz'))
        })
    return data_dicts


def get_loader(args):
    train_transform = get_train_transform(args)
    val_transform = get_val_transform(args)

    dl = MyDataLoader(
        get_data_dicts,
        train_transform,
        val_transform,
        args
    )

    return dl.get_loader()


def get_infer_data(data_dict, args):
    keys = data_dict.keys()
    inf_transform = get_inf_transform(keys, args)
    data = inf_transform(data_dict)
    return data


def get_infer_loader(keys, args):
    data_dicts = [{'image': args.img_pth, 'label': args.lbl_pth}]
    inf_transform = get_inf_transform(keys, args)
    inf_loader = get_dl(
        files=data_dicts,
        transform=inf_transform,
        shuffle=False,
        batch_size=args.batch_size,
        args=args
    )
    return inf_loader
