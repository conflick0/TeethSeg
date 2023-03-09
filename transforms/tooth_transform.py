from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    RandRotate90d,
    ToTensord,
    LabelToMaskd
)
from transforms.transform_utils import LabelToBinaryLabeld

def scale_intensity(args):
    if args.scale_intensity_type == 'percent':
        print('use scale intensity by percent')
        return ScaleIntensityRangePercentilesd(
            keys=["image"], 
            lower=1, 
            upper=99, 
            b_min=0.0,
            b_max=1.0,
            clip=True
        )
    elif args.scale_intensity_type == 'range':
        print('use scale intensity by range')
        return ScaleIntensityRanged(
            keys=["image"], 
            a_min=args.a_min, 
            a_max=args.a_max, 
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        )
    else:
        raise ValueError(f'Invalid scale intensity type: {args.scale_intensity_type}')


def get_train_transform(args):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            LabelToBinaryLabeld(keys=["label"]),
            # LabelToMaskd(keys=["label"], select_labels=[i for i in range(1, args.select_label_num)]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            scale_intensity(args),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=args.rand_flipd_prob,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=args.rand_flipd_prob,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=args.rand_flipd_prob,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=args.rand_rotate90d_prob,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=args.rand_shift_intensityd_prob,
            ),
            ToTensord(keys=["image", "label"])
        ]
    )


def get_val_transform(args):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            LabelToBinaryLabeld(keys=["label"]),
            # LabelToMaskd(keys=["label"], select_labels=[i for i in range(1, args.select_label_num)]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            scale_intensity(args),
            ToTensord(keys=["image", "label"])
        ]
    )


def get_inf_transform(keys, args):
    if len(keys) == 2:
        # image and label
        mode = ("bilinear", "nearest")
    elif len(keys) == 3:
        # image and mutiple label
        mode = ("bilinear", "nearest", "nearest")
    else:
        # image
        mode = ("bilinear")
        
    return Compose(
        [
            LoadImaged(keys=keys),
            AddChanneld(keys=keys),
            LabelToBinaryLabeld(keys=["label"]),
            # LabelToMaskd(keys=["label"], select_labels=[i for i in range(1, args.select_label_num)]),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=mode,
            ),
            scale_intensity(args),
            AddChanneld(keys=keys),
            ToTensord(keys=keys)
        ]
    )
