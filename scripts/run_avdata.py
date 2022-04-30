from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

from avdata import AgentBatch, AgentType, UnifiedDataset


if __name__ == "__main__":

    dataset = UnifiedDataset(
        desired_data=["nusc"],
        centric="agent",
        history_sec=(1.5, 1.5),
        future_sec=(5.0, 5.0),
        # rebuild_maps=True,
        # rebuild_cache=True,
        only_types=[AgentType.VEHICLE],
        data_dirs= {
            "nusc": "~/workspace/lfs/nuscenes/",
            "nusc_mini": "~/workspace/lfs/nuscenes/",
            "lyft_sample": "~/datasets/lyft/scenes/sample.zarr",
        },
        # incl_robot_future=True,
        agent_interaction_distances=defaultdict(lambda: 30.0),
        incl_map=True,
        map_params={"px_per_m": 2, "map_size_px": 224},
        num_workers=4,
        verbose=True,
    )
    #
    # print(f"# Data Samples: {len(dataset):,}")
    #
    dataloader = DataLoader(
        dataset,
        batch_size=100,
        shuffle=True,
        collate_fn=dataset.get_collate_fn(return_dict=True),
        num_workers=4,
        drop_last=True
    )

    batch: AgentBatch
    for batch in tqdm(dataloader):
        from IPython import embed; embed()
        pass