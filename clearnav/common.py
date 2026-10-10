import numpy as np
import torch
import torch as th
from typing import Optional, Tuple, List
from scipy.spatial.transform import Rotation as Rotation
from enum import Enum


class ExitCode(Enum):
    SUCCESS = 0
    ERROR = 1
    EARLY_STOP = 2
    NOT_FOUND = 3
    TIMEOUT = 4
    KEYBOARD_INTERRUPT = 5


def std_to_habitat(
    std_pos: Optional[torch.Tensor] = None,
    std_ori: Optional[torch.Tensor] = None,
    format="enu",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """_summary_
        axes transformation, from std to habitat-sim

    Args:
        std_pos (_type_): _description_
        std_ori (_type_): _description_
        format (str, optional): _description_. Defaults to "enu".

    Returns:
        _type_: _description_
    """
    assert format in ["enu"]

    if std_ori is None:
        hab_ori = None
    else:
        hab_ori = std_ori.clone().detach().cpu().numpy() @ np.array(
            [[1, 0, 0, 0], [0, 0, 0, -1], [0, -1, 0, 0], [0, 0, 1, 0]]
        )

    if std_pos is None:
        hab_pos = None
    else:
        if len(std_pos.shape) == 1:
            hab_pos = (
                std_pos.clone().detach().cpu().unsqueeze(0).numpy()
                @ np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
            ).squeeze()
        elif std_pos.shape[1] == 3:
            hab_pos = std_pos.clone().detach().cpu().numpy() @ np.array(
                [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
            )
        else:
            raise ValueError("std_pos shape error")

    return hab_pos, hab_ori


def habitat_to_std(
    habitat_pos: Optional[np.ndarray] = None,
    habitat_ori: Optional[np.ndarray] = None,
    format="enu",
):
    """_summary_
        axes transformation, from habitat-sim to std

    Args:
        habitat_pos (_type_): _description_
        habitat_ori (_type_): _description_
        format (str, optional): _description_. Defaults to "enu".

    Returns:
        _type_: _description_
    """
    # habitat_pos, habitat_ori = np.atleast_2d(habitat_pos), np.atleast_2d(habitat_ori)
    assert format in ["enu"]

    if habitat_pos is None:
        std_pos = None
    else:
        # assert habitat_pos.shape[1] == 3
        std_pos = th.as_tensor(
            np.atleast_2d(habitat_pos) @ np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
            dtype=th.float32,
        )
        # if len(habitat_pos.shape) == 1:
        #     std_pos = habitat_pos

    if habitat_ori is None:
        std_ori = None
    else:
        # assert habitat_ori.shape[1] == 4
        std_ori = th.from_numpy(
            np.atleast_2d(habitat_ori)
            @ np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1], [0, -1, 0, 0]])
        )
    return std_pos, std_ori


def obs_list2array(obs_dict: List, row: int, column: int):
    obs_indice = 0
    obs_array = []
    for i in range(column):
        obs_row = []
        for j in range(row):
            obs_row.append(obs_dict[obs_indice]["depth"])
            obs_indice += 1
        obs_array.append(np.hstack(obs_row))
    return np.vstack(obs_array)


def rgba2rgb(image):
    if isinstance(image, List):
        return [rgba2rgb(img) for img in image]
    else:
        return image[:, :, :3]


def observation_to_device(obs, device):
    return {k: v.to(device) for k, v in obs.items()}
