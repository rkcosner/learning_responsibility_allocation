from tbsim.configs.base_config import TrainConfig, EnvConfig, AlgoConfig


class L5KitTrainConfig(TrainConfig):
    def __init__(self):
        super(L5KitTrainConfig, self).__init__()

        self.dataset_path = "path-to-dataset"
        self.dataset_valid_key = "scenes/train.zarr"
        self.dataset_train_key = "scenes/validate.zarr"
        self.dataset_mata_key = "meta.json"


class L5KitEnvConfig(EnvConfig):
    def __init__(self):
        super(L5KitEnvConfig, self).__init__()

        # raster image size [pixels]
        self.rasterizer.raster_size = (224, 224)

        # raster's spatial resolution [meters per pixel]: the size in the real world one pixel corresponds to.
        self.rasterizer.pixel_size = (0.5, 0.5)

        # From 0 to 1 per axis, [0.5,0.5] would show the ego centered in the image.
        self.rasterizer.ego_center = (0.25, 0.5)

        self.rasterizer.map_type = "py_semantic"

        # the keys are relative to the dataset environment variable
        self.rasterizer.satellite_map_key = "aerial_map/aerial_map.png"
        self.rasterizer.semantic_map_key = "semantic_map/semantic_map.pb"

        # e.g. 0.0 include every obstacle, 0.5 show those obstacles with >0.5 probability of being
        # one of the classes we care about (cars, bikes, peds, etc.), >=1.0 filter all other agents.
        self.rasterizer.filter_agents_threshold = 0.5

        # whether to completely disable traffic light faces in the semantic rasterizer
        self.rasterizer.disable_traffic_light_faces = False

        # When set to True, the rasterizer will set the raster origin at bottom left,
        # i.e. vehicles are driving on the right side of the road.
        # With this change, the vertical flipping on the raster used in the visualization code is no longer needed.
        # Set it to False for models trained before v1.1.0-25-g3c517f0 (December 2020).
        # In that case visualisation will be flipped (we've removed the flip there) but the model's input will be correct.
        self.rasterizer.set_origin_to_bottom =True


class L5RasterizedPlanningConfig(AlgoConfig):
    def __init__(self):
        super(L5RasterizedPlanningConfig, self).__init__()
        self.name = "l5_rasterized"
        self.model_architecture = "resnet50"
        self.history_num_frames = 5
        self.future_num_frames = 50
        self.step_time = 0.1
        self.render_ego_history = False
