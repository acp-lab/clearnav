import random
import glob
import os
from torch.utils.data import Dataset


class SimpleDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.paths = dataset.paths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.paths)

    @property
    def len(self):
        return len(self.paths)

    def _create_iter_index(self):
        self._select_index = list(range(self.len))
        random.shuffle(self._select_index) if self.shuffle else None
        self._batches_num = self.len // self.batch_size + (
            self.len % self.batch_size != 0
        )
        self._select_index_batches = []
        for index in range(self._batches_num):
            start = self.batch_size * index
            end = (
                self.batch_size * (index + 1)
                if self.batch_size * (index + 1) <= self.len
                else None
            )
            self._select_index_batches.append(self._select_index[start:end])

    def __iter__(self):
        self._create_iter_index()
        self._current_batch_index = 0
        return self

    def __next__(self):
        if self._current_batch_index >= self._batches_num:
            raise StopIteration
        paths_output_list = []
        for index in self._select_index_batches[self._current_batch_index]:
            paths_output_list.append(self.paths[index])
        self._current_batch_index += 1
        return paths_output_list

    def _reset(self):
        self._current_batch_index = 0


class ChildrenPathDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.paths = self._load_scene_path()

    def _load_scene_path(self):
        key = "*.scene_instance.json"

        if os.path.isdir(self.root_path):
            glb_files = []
            for root, dirs, files in os.walk(self.root_path):
                file_path = glob.glob(os.path.join(root, key))
                glb_files.extend(file_path)
        else:
            basename = os.path.basename(self.root_path)
            directory = os.path.dirname(self.root_path)
            file_paths = os.listdir(directory)
            glb_files = [
                os.path.join(directory, filename)
                for filename in file_paths
                if basename in filename
            ]

        glb_files = sorted(glb_files)
        return glb_files

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, indice):
        return self.paths[indice]
