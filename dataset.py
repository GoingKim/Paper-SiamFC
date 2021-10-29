##
import numpy as np
import cv2

import torch


##
class Dataset(torch.utils.data.Dataset):
    def __init__(self, seqs, transform=None):

        # When inherited from 'Dataset parent class',
        # 'super().__init__()' makes child class avoid overriding. (*Important)
        super().__init__()

        # In this code, seqs is GOT-10k Dataset.
        # code> seqs = GOT10k(root_dir, subset='train', return_meta=True)
        self.seqs = seqs

        # permutation=순열, np.random.permutation(n) makes a random number list(n)
        # ex> [2, 1, 6, 23, 3, 5, ...]
        self.indices = np.random.permutation(len(seqs))

        # getattr(np,'array') is same to np.array method. (call method)
        # but string, ' ', can be used member variable and class etc. It is usefull to make code clean.
        self.return_meta = getattr(seqs, 'return_meta', False)

        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # previous code> index = self.indices[index % len(self.indices)]
        # I think '% len(self.indices)' is useless.
        index = self.indices[index]

        # get filename lists and annotations
        # seqs are folders  ex> GOT-10k_Train_XXXXXXXX
        # seqs[i] is a folder including all object images.
        # 'groundtruth.txt' file has 4 values in list form. ex> (x0, y0, width, height)

        # train/valid -> return_meta=True, test -> return_meta=False
        # return_meta=True, False라는 GOT-10k Toolkit의 변수에 따라, 데이터 셋의 학습/라벨 정보를 받아오는 방법이 달라진다.
        # meta variable get the info from 'meta_info.ini' file.
        # and meta.get call the info, cover(0)/absence(0~?)/cut_by_image(0) in np.array form.
        if self.return_meta:
            img_files, label, meta = self.seqs[index]

            # attributes = ['cover', 'absence', 'cut_by_image']
            # When train, meta data shows attributes for data.
            vis_ratios = meta.get('cover', None)
        else:
            img_files, label = self.seqs[index][:2]
            vis_ratios = None

        # filter out noisy frames
        # Train/Valid 폴더 안에 있는 모든 사진들을 학습에 사용하지 않는다.
        # 어떤 Treshhold 값에 미치지 못하는 데이터들은 학습에서 제외하기 위해 필터 실행한다.
        # 첫 번째 argument로 imge_files[0]를 받는 이유는 첫 번째 프레임을 사용하기 위한 것이 아니라,
        # filter condition을 사용하기 위해 img size 값을 사용하는데 그 값을 참조하기 위해 임의의 사진 한 장만 불러오는 것이다.
        # 그러므로 label에는 하나의 폴더 안에 있는 모든 bbox 좌표 값이 리스트 형태로 입력된다.
        val_indices = self._filter(cv2.imread(img_files[0], cv2.IMREAD_COLOR), label, vis_ratios)

        # val_indices value가 2보다 작다면(한쌍으로 사용할 수 없다면) 노이즈가 많다고 판단하고 그 폴더에 데이터는 학습에 사용하지 않는다.
        # seqs[1] 일 때는 val_indices에 빈 리스트가 리턴되었고 seqs[0]일 때는 길이가 69인 리스트(value 0~109 오름차순)가 리턴되었다.
        # 폴더 안에 있는 사진들로 2장으로 한 쌍을 구성하지 못하는 경우, seqs에서 다른 폴더를 학습에 사용한다.
        if len(val_indices) < 2:
            index = np.random.choice(len(self))
            return self.__getitem__(index)

        # sample a frame pair
        # _sampler_pair 함수를 통해
        # 하나의 폴더에서 exampler_z, serach image_x 각각 한 장씩 랜덤으로 2장의 사진을 선택한다.
        # 학습에서는 1 epoch에 한 폴더에서 2장의 사진만 사용하는 듯 하다.
        # ex> seq[0]에서 [0, ..., 109] 인덱스 리스트에서 np.random.choice를 통해 [z, x]형태 2개의 index 골라낸다.
        # index는 random.choice 이므로 매 에폭마다 바뀐다.
        rand_z, rand_x = self._sample_pair(val_indices)

        z = cv2.imread(img_files[rand_z], cv2.IMREAD_COLOR)
        x = cv2.imread(img_files[rand_x], cv2.IMREAD_COLOR)
        z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        box_z = label[rand_z]
        box_x = label[rand_x]

        # (img_file_name, img_file_name, label, label)
        input = (z, x, box_z, box_x)

        if self.transform:
            input = self.transform(*input)

        return input

    def _sample_pair(self, indices):
        n = len(indices)
        assert n > 0

        if n == 1:
            return indices[0], indices[0]
        elif n == 2:
            return indices[0], indices[1]
        else:
            for i in range(100):
                rand_z, rand_x = np.sort(
                    np.random.choice(indices, 2, replace=False))
                if rand_x - rand_z < 100:
                    break
            else:
                rand_z = np.random.choice(indices)
                rand_x = rand_z

            return rand_z, rand_x

    def _filter(self, img0, anno, vis_ratios=None):
        size = np.array(img0.shape[1::-1])[np.newaxis, :]
        areas = anno[:, 2] * anno[:, 3]

        # acceptance conditions
        c1 = areas >= 20
        c2 = np.all(anno[:, 2:] >= 20, axis=1)
        c3 = np.all(anno[:, 2:] <= 500, axis=1)
        c4 = np.all((anno[:, 2:] / size) >= 0.01, axis=1)
        c5 = np.all((anno[:, 2:] / size) <= 0.5, axis=1)
        c6 = (anno[:, 2] / np.maximum(1, anno[:, 3])) >= 0.25
        c7 = (anno[:, 2] / np.maximum(1, anno[:, 3])) <= 4
        if vis_ratios is not None:
            c8 = (vis_ratios > max(1, vis_ratios.max() * 0.3))
        else:
            c8 = np.ones_like(c1)

        mask = np.logical_and.reduce(
            (c1, c2, c3, c4, c5, c6, c7, c8))
        val_indices = np.where(mask)[0]

        return val_indices