import bisect
from functools import partial
from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset

from l5kit.data import ChunkedDataset, get_frames_slice_from_scenes
from l5kit.dataset.utils import convert_str_to_fixed_length_tensor
from l5kit.kinematic import Perturbation
from l5kit.rasterization import Rasterizer, RenderContext
from l5kit.sampling.agent_sampling import generate_agent_sample
from l5kit.sampling.agent_sampling_vectorized import generate_agent_sample_vectorized
from tbsim.external.agent_sampling_mixed import generate_agent_sample_mixed
from l5kit.vectorization.vectorizer import Vectorizer
from l5kit.dataset.ego import BaseEgoDataset


class EgoDatasetMixed(BaseEgoDataset):
    def __init__(
        self,
        cfg: dict,
        zarr_dataset: ChunkedDataset,
        vectorizer: Vectorizer,
        rasterizer: Rasterizer,
        perturbation: Optional[Perturbation] = None,
    ):
        """
        Get a PyTorch dataset object that can be used to train DNNs with vectorized input

        Args:
            cfg (dict): configuration file
            zarr_dataset (ChunkedDataset): the raw zarr dataset
            vectorizer (Vectorizer): a object that supports vectorization around an AV
            perturbation (Optional[Perturbation]): an object that takes care of applying trajectory perturbations.
        None if not desired
        """
        self.perturbation = perturbation
        self.vectorizer = vectorizer
        self.rasterizer = rasterizer
        super().__init__(cfg, zarr_dataset)

    def _get_sample_function(self) -> Callable[..., dict]:
        render_context = RenderContext(
            raster_size_px=np.array(self.cfg["raster_params"]["raster_size"]),
            pixel_size_m=np.array(self.cfg["raster_params"]["pixel_size"]),
            center_in_raster_ratio=np.array(self.cfg["raster_params"]["ego_center"]),
            set_origin_to_bottom=self.cfg["raster_params"]["set_origin_to_bottom"],
        )
        return partial(
            generate_agent_sample_mixed,
            render_context=render_context,
            history_num_frames_ego=self.cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=self.cfg["model_params"][
                "history_num_frames_agents"
            ],
            future_num_frames=self.cfg["model_params"]["future_num_frames"],
            step_time=self.cfg["model_params"]["step_time"],
            filter_agents_threshold=self.cfg["raster_params"][
                "filter_agents_threshold"
            ],
            perturbation=self.perturbation,
            vectorizer=self.vectorizer,
            rasterizer=self.rasterizer,
            vectorize_lane=self.cfg["data_generation_params"]["vectorize_lane"],
        )

    def get_scene_dataset(self, scene_index: int) -> "EgoDatasetMixed":
        dataset = self.dataset.get_scene_dataset(scene_index)
        return EgoDatasetMixed(
            self.cfg,
            dataset,
            self.vectorizer,
            self.rasterizer,
            self.perturbation,
        )

    def get_frame(
            self, scene_index: int, state_index: int, track_id: Optional[int] = None
    ) -> dict:
        data = super().get_frame(scene_index, state_index, track_id=track_id)
        # TODO (@lberg): this should not be here but in the rasterizer
        data["image"] = data["image"].transpose(2, 0, 1)  # 0,1,C -> C,0,1
        return data
