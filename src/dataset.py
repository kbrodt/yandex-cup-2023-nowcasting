import random
from datetime import datetime

import cv2
import h5py
import numpy as np
import torch.utils.data as data


def rotate_image(image, angle, rot_pnt, scale=1):
    rot_mat = cv2.getRotationMatrix2D(rot_pnt, angle, scale)
    result = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return result


def train_aug(imgs, mask, target):
    # imgs: [t, c, h, w]
    # mask: [t, ]
    # target: [t, c, h, w]

    if random.random() > 0.5:  # horizontal flip
        imgs = imgs[..., ::-1]
        target = target[..., ::-1]

    k = random.randrange(4)
    if k > 0:  # rotate90
        imgs = np.rot90(imgs, k=k, axes=(-2, -1))
        target = np.rot90(target, k=k, axes=(-2, -1))

    if random.random() > 0.3:  # scale-rotate
        _d = int(imgs.shape[2] * 0.1)  # 0.4)
        rot_pnt = (imgs.shape[2] // 2 + random.randint(-_d, _d), imgs.shape[3] // 2 + random.randint(-_d, _d))
        angle = random.randint(0, 90) - 45

        if (angle != 0):# or (scale != 1):
            t = len(imgs)  # t, c, h, w
            imgs = np.concatenate(imgs, axis=0)  # t*c, h, w
            imgs = np.transpose(imgs, (1, 2, 0))  # h, w, t*c
            imgs = rotate_image(imgs, angle, rot_pnt)
            imgs = np.transpose(imgs, (2, 0, 1))  # t*c, h, w
            imgs = np.reshape(imgs, (t, -1, imgs.shape[1], imgs.shape[2]))  # t, c, h, w

            t = len(target)  # t, c, h, w
            target = np.concatenate(target, axis=0)  # t*c, h, w
            target = np.transpose(target, (1, 2, 0))  # h, w, t*c
            target = rotate_image(target, angle, rot_pnt)
            target = np.transpose(target, (2, 0, 1))  # t*c, h, w
            target = np.reshape(target, (t, -1, target.shape[1], target.shape[2]))  # t, c, h, w

    return imgs.copy(), mask, target.copy()


class RadarDataset(data.Dataset):
    def __init__(self, list_of_files, in_seq_len=4, out_seq_len=12, mode="overlap", with_time=False, is_train=False):
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.seq_len = in_seq_len + out_seq_len
        self.with_time = with_time
        self.is_train = is_train
        self.__prepare_timestamps_mapping(list_of_files)
        self.__prepare_sequences(mode)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        refs = []
        evs = []
        data = []
        mask = []
        months = []
        for timestamp in self.sequences[index]:
            with h5py.File(self.timestamp_to_file[timestamp]) as d:
                arr = np.array(d[timestamp]["intensity"])#, dtype="float32")

                dt_obj = datetime.fromtimestamp(int(timestamp))
                month = float(dt_obj.month) / 12
                months.append(month)

                data.append(arr)

                if len(evs) < self.in_seq_len:
                    ev = np.array(d[timestamp]["events"])
                    evs.append(ev)
                    ref = np.array(d[timestamp]["reflectivity"][:4])
                    refs.append(ref)

                mask.append(not (arr > 0).any())

        evs = np.expand_dims(evs, axis=1)   # [t, 1, h, w]
        evs[evs == -1e6] = -1
        evs[evs == -2e6] = -2
        evs += 1
        evs[evs != -1] /= 20

        refs = np.concatenate(refs, axis=0)   # [2*t, h, w]
        refs = np.expand_dims(refs, axis=1)   # [2*t, 1, h, w]
        refs[refs == -1e6] = -1
        refs[refs == -2e6] = -2
        refs += 1
        refs[refs != -1] /= 71

        data = np.expand_dims(data, axis=1)   # [t, 1, h, w]
        inputs = data[:self.in_seq_len]
        inputs[inputs < 0] = 0
        m = inputs.max((1, 2, 3), keepdims=True)
        m[m == 0] = 1
        inputs /= m

        inputs = np.concatenate(
            [
                inputs,  # 4
                evs,     # 4
                refs,    # 8
            ],
            axis=0,
        )  # [2t, 1, h, w]

        targets = data[self.in_seq_len:]
        targets[targets == -1e6] = 0
        targets[targets == -2e6] = -1

        mask = np.array(mask, dtype="bool")[:self.in_seq_len]

        month = np.mean(months).astype("float32")

        if self.is_train:
            inputs, mask, targets = train_aug(inputs, mask, targets)

        if self.with_time:
            return (inputs, month, mask, self.sequences[index][-1]), targets
        else:
            return inputs, month, mask, targets

    def __prepare_timestamps_mapping(self, list_of_files):
        self.timestamp_to_file = {}
        for filename in list_of_files:
            with h5py.File(filename) as d:
                self.timestamp_to_file = {
                    **self.timestamp_to_file,
                    **dict(map(lambda x: (x, filename), d.keys()))
                }

    def __prepare_sequences(self, mode):
        timestamps = np.unique(sorted(self.timestamp_to_file.keys()))
        if mode == 'sequentially':
            self.sequences = [
                timestamps[index * self.seq_len: (index + 1) * self.seq_len]
                for index in range(len(timestamps) // self.seq_len)
            ]
        elif mode == 'overlap':
            self.sequences = [
                timestamps[index: index + self.seq_len]
                for index in range(len(timestamps) - self.seq_len + 1)
            ]
        else:
            raise Exception(f'Unknown mode {mode}')
        self.sequences = list(filter(
            lambda x: int(x[-1]) - int(x[0]) == (self.seq_len - 1) * 600,
            self.sequences
        ))
