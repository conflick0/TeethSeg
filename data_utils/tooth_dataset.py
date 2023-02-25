import os

from data_utils.data_loader import MyDataLoader, get_dl
from transforms.tooth_transform import get_train_transform, get_val_transform, get_inf_transform


def get_data_dicts(data_dir):
    # img and lbl dir
    img_dir = os.path.join(data_dir, 'image')
    lbl_dir = os.path.join(data_dir, 'label')

    # file names
    file_names = sorted(os.listdir(img_dir))

    # data dicts
    data_dicts = []
    for file_name in file_names:
        data_dicts.append({
            "image": os.path.join(os.path.join(img_dir, file_name)),
            "label": os.path.join(os.path.join(lbl_dir, file_name))
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
