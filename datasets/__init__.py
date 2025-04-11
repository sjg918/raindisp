from .kitti_dataset import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .drivingstereo_dataset import DrivingStereoDataset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
}
