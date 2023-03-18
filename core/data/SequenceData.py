from torch.utils.data import Dataset
import numpy as np
import h5py
import os

from utils.registery import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SequenceData(Dataset):
    def __init__(self,
                 feat_root,
                 label_root,
                 feat_dict,
                 seq_len,
                 task,
                 pad_mode='repeat_last'):
        """SequenceData

        Args:
            feat_root (str): feat root path
            label_root (str): label root path
            feat_dict (dict): feat dict in which key is `feat_name`, value is `feat_dim`
            seq_len (int): sequence length
            task (str): `va`, `expr` or `au`
            pad_mode (str): pad mode, here just implemented `repeat_last` (default: 'repeat_last')
        """
        self.feat_root = feat_root
        self.label_root = label_root
        self.feat_dict = feat_dict
        self.seq_len = seq_len
        self.task = task
        self.pad_mode = pad_mode
        self.feat_map = dict()
        self.sequence_list, self.label_list = self.make_sequence()

    def get_txt_contents(self, path):
        """get txt annotation contents, and return a dict which key is `frame_id`(aligned with 05d), value is `annotation`.
        In task `va`:, content is like {'00001': [0.1, 0.2], ...}
        In task `expr`:, content is like {'00001': 1, ...}
        In task `au`:, content is like {'00001': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], ...}

        Args:
            path (str): txt file path

        Returns:
            content (dict)
        """
        with open(path, 'r') as f:
            content = dict()
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                if self.task == 'va':
                    value_list = [float(value)
                                  for value in line.strip('\n').split(',')]
                    content[f'{i :05d}'] = value_list
                elif self.task == 'expr':
                    value_list = int(line.strip('\n'))
                    content[f'{i :05d}'] = value_list
                elif self.task == 'au':
                    value_list = [float(value)
                                  for value in line.strip('\n').split(',')]
                    content[f'{i :05d}'] = value_list

        return content

    def __filter_invalid_annotations(self, label_dict, video_name):
        """filter invalid annotation like `-5` in va, `-1` in expr and au.
        In some case, annotations are given by organizer, but images are not provided (in task `va` and `expr`), so ... just ignore that

        Args:
            label_dict (dict): label dict generate by :method `get_txt_contents`
            video_name (str): video name string without ext

        Returns:
            returned_label_dict (dict): dict organized like `label_dict`, but is filtered.
        """
        self.open_h5()

        returned_label_dict = label_dict.copy()
        if self.task == 'va':
            for seq_id in label_dict.keys():
                if (-5 in label_dict[seq_id]):
                    returned_label_dict.pop(seq_id)
        elif self.task == 'expr':
            for seq_id in label_dict.keys():
                if (-1 == label_dict[seq_id]):
                    returned_label_dict.pop(seq_id)
        elif self.task == 'au':
            for seq_id in label_dict.keys():
                if -1 in label_dict[seq_id]:
                    returned_label_dict.pop(seq_id)

        self.close_h5()

        return returned_label_dict

    def get_video_list(self):
        """get video list

        Returns:
            video_list (list): video list in `task`
        """
        video_list = [x.split('.')[0]
                      for x in sorted(os.listdir(self.label_root))]

        return video_list

    def make_sequence_id_list(self, label_dict, video_name):
        """make sequence id list and sequence label list, it's upon to :attribute `pad_mode`

        Args:
            label_dict (dict): label dict generate by :method `get_txt_contents`
            video_name (str): video name string without ext

        Returns:
            sequence_id_list (list): 2-d sequence id list like [[1, 2, 3, ..., 127, 127], [...], ...]
            sequence_label_list (list): sequence label list
                                        in task `va` like: [[[0.0, 0.1], [0.0, 0.1], ...], [...], ...]
                                        in task `expr` like: [[0, 4, ...], [...], ...]
                                        in task `au` like: [[[0, 0, 1, 0, ...], ...], [...], ...]

        """
        label_dict = self.__filter_invalid_annotations(label_dict, video_name)
        sequence_id_list = list(label_dict.keys())
        sequence_label_list = list()
        if self.pad_mode == 'repeat_last':
            sequence_id_list = [sequence_id_list[i: i + self.seq_len]
                                for i in range(0, len(sequence_id_list), self.seq_len)]

            for seq in sequence_id_list:
                for i in range(len(seq)):
                    seq[i] = video_name + '/' + str(seq[i])

            for i in range(len(sequence_id_list)):
                if len(sequence_id_list[i]) < self.seq_len:
                    pad_list = sequence_id_list[i]
                    while (len(pad_list) < self.seq_len):
                        pad_list.append(pad_list[-1])
                    sequence_id_list[i] = pad_list

            for sequence_id in sequence_id_list:
                sequence_label_list.append(
                    [label_dict[k.split('/')[-1]] for k in sequence_id])

            return sequence_id_list, sequence_label_list

    def make_sequence(self):
        """make sequence

        Returns:
            seq_id_list (list): explained in :method `make_sequence_id_list`
            seq_label_list (list): explained in :method `make_sequence_id_list`
        """
        seq_id_list = list()
        seq_label_list = list()
        video_list = self.get_video_list()
        for video in video_list:
            txt_path = os.path.join(self.label_root, video + '.txt')
            label_dict = self.get_txt_contents(txt_path)
            crt_id_list, crt_label_list = self.make_sequence_id_list(
                label_dict, video)
            seq_id_list += crt_id_list
            seq_label_list += crt_label_list

        return seq_id_list, seq_label_list

    def open_h5(self):
        for feat_name in self.feat_dict.keys():
            self.feat_map[feat_name] = h5py.File(
                os.path.join(self.feat_root, feat_name + '.h5'), 'r')

    def close_h5(self):
        for feat_name in self.feat_dict.keys():
            self.feat_map[feat_name].close()

    def __len__(self):
        return len(self.sequence_list)

    def __getitem__(self, idx):
        self.open_h5()
        seq_list = self.sequence_list[idx]
        lb_list = self.label_list[idx]
        assert len(seq_list) == self.seq_len == len(lb_list)
        seq_feat = list()
        seq_label = list()
        for seq_name, label in zip(seq_list, lb_list):
            feat = list()
            for feat_name in self.feat_map.keys():
                audio = False
                if feat_name in ['ecapatdnn', 'wav2vec', 'wav2vec_emotion', 'fbank', 'hubert', 'vggish']:
                    audio = True
                    name = seq_name.replace('_right', '').replace('_left', '')
                else:
                    name = seq_name

                try:
                    crt_feat = np.asarray(self.feat_map[feat_name][name])
                except Exception:
                    if audio:
                        video_name = name.split('/')[0]
                        frame_id = list(
                            self.feat_map[feat_name][video_name].keys())[-1]
                        crt_feat = np.asarray(
                            self.feat_map[feat_name][f'{video_name}/{frame_id}'])
                    else:
                        video_name, frame_id = name.split('/')
                        total_frame_num = len(list(self.feat_map[feat_name][video_name].keys()))

                        for crt_id in range(1, total_frame_num):
                            crt_id_upper = int(frame_id) + crt_id
                            crt_id_lower = int(frame_id) - crt_id
                            upper_key = f'{crt_id_upper :05d}'
                            lower_key = f'{crt_id_lower :05d}'
                            if upper_key in self.feat_map[feat_name][video_name].keys():
                                crt_feat = np.asarray(self.feat_map[feat_name][f'{video_name}/{upper_key}'])
                                break
                            elif lower_key in self.feat_map[feat_name][video_name].keys():
                                crt_feat = np.asarray(self.feat_map[feat_name][f'{video_name}/{lower_key}'])
                                break
                feat.append(crt_feat)
            feat = np.concatenate(feat, axis=-1)
            seq_feat.append(feat)
            seq_label.append(np.asarray(label))

        seq_feat = np.asarray(seq_feat)
        seq_label = np.asarray(seq_label)

        self.close_h5()

        return seq_feat, seq_label, seq_list
