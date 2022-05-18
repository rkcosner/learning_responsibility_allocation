import json
import argparse
import numpy as np
import os
from pprint import pprint
import torch
import h5py
from avdata.simulation.sim_stats import calc_stats
import tbsim.utils.tensor_utils as TensorUtils
import pathlib
from pyemd import emd


def parse(args):
    rjson = json.load(open(os.path.join(args.results_dir, "stats.json"), "r"))
    cfg = json.load(open(os.path.join(args.results_dir, "config.json"), "r"))

    results = dict()
    for k in rjson:
        if k != "scene_index":
            if args.num_scenes is None:
                rnum = np.mean(rjson[k])
                print("{} = {}".format(k, np.mean(rjson[k])))
            else:
                rnum = np.mean(rjson[k][:args.num_scenes])
                print("{} = {}".format(k, rnum))
            results[k] = rnum

    hist_stats_fn = os.path.join(args.results_dir, "hist_stats.json")
    if not os.path.exists(hist_stats_fn):
        compute_and_save_stats(os.path.join(args.results_dir, "data.hdf5"))

    hjson = json.load(open(hist_stats_fn, "r"))
    if cfg["env"] == "l5kit":
        gt_hjson = json.load(open("/home/danfeix/workspace/tbsim/results/l5kit/GroundTruth/hist_stats.json", "r"))
    elif cfg["env"] == "nusc":
        gt_hjson = json.load(open("/home/danfeix/workspace/tbsim/results/nusc/GroundTruth/hist_stats.json", "r"))

    for k in gt_hjson["stats"]:
        results["{}_dist".format(k)] = calc_hist_distance(
            hist1=np.array(gt_hjson["stats"][k]),
            hist2=np.array(hjson["stats"][k]),
            bin_edges=np.array(gt_hjson["ticks"][k][1:])
        )


    print("num_scenes: {}".format(len(rjson["scene_index"])))
    ade = results["ade"] if "ade" in results else results["ADE"]
    fde = results["fde"] if "fde" in results else results["FDE"]

    pprint(cfg["ckpt"])
    results_str = [
        ade,
        fde,
        results["all_failure_any"] * 100,
        results["all_failure_coll"] * 100,
        results["all_failure_offroad"] * 100,
        results["all_diversity"],
        results["all_coverage_success"],
        results["all_coverage_total"],
        results["all_collision_rate_coll_any"] * 100,
        results["all_collision_rate_CollisionType.REAR"] * 100,
        results["all_collision_rate_CollisionType.FRONT"] * 100,
        results["all_collision_rate_CollisionType.SIDE"] * 100,
        results["all_off_road_rate_rate"] * 100,
        results["velocity_dist"],
        results["lon_accel_dist"],
        results["lat_accel_dist"],
        results["jerk_dist"]
    ]

    results_str = ["{:.3f}".format(r) for r in results_str]

    print(",".join(results_str))


def calc_hist_distance(hist1, hist2, bin_edges):
    bins = np.array(bin_edges)
    bins_dist = np.abs(bins[:, None] - bins[None, :])
    hist_dist = emd(hist1, hist2, bins_dist)
    return hist_dist


def compute_and_save_stats(h5_path):
    h5f = h5py.File(h5_path, "r")
    bins = {
        "velocity": torch.linspace(0, 30, 21),
        "lon_accel": torch.linspace(0, 10, 21),
        "lat_accel": torch.linspace(0, 10, 21),
        "jerk": torch.linspace(0, 20, 21),
    }

    sim_stats = dict()
    # gt_stats = dict()
    ticks = None

    for i, scene_index in enumerate(h5f.keys()):
        if i % 10 == 0:
            print(i)
        scene_data = h5f[scene_index]
        sim_pos = scene_data["centroid"]
        sim_yaw = scene_data["yaw"][:][:, None]

        # gt_pos = scene_data["gt_centroid"]
        # gt_yaw = scene_data["gt_yaw"][:][:, None]

        sim = calc_stats(positions=torch.Tensor(sim_pos), heading=torch.Tensor(sim_yaw), dt=0.1, bins=bins)
        # gt = calc_stats(positions=torch.Tensor(gt_pos), heading=torch.Tensor(gt_yaw), dt=0.1, bins=bins)

        for k in sim:
            if k not in sim_stats:
                sim_stats[k] = sim[k].hist.long()
            else:
                sim_stats[k] += sim[k].hist.long()

        # for k in gt:
        #     if k not in gt_stats:
        #         gt_stats[k] = gt[k].hist.long()
        #     else:
        #         gt_stats[k] += gt[k].hist.long()

        if ticks is None:
            ticks = dict()
            for k in sim:
                ticks[k] = sim[k].bin_edges

    # for k in gt_stats:
    #     gt_stats[k] = TensorUtils.to_numpy(gt_stats[k] / len(h5f.keys())).tolist()

    for k in sim_stats:
        sim_stats[k] = TensorUtils.to_numpy(sim_stats[k] / len(h5f.keys())).tolist()
    for k in ticks:
        ticks[k] = TensorUtils.to_numpy(ticks[k]).tolist()

    results_path = pathlib.Path(h5_path).parent.resolve()
    output_file = os.path.join(results_path, "hist_stats.json")
    json.dump({"stats": sim_stats, "ticks": ticks}, open(output_file, "w+"), indent=4)
    print("results dumped to {}".format(output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="A directory of results files (including config.json and stats.json)"
    )

    parser.add_argument(
        "--num_scenes",
        type=int,
        default=None
    )

    args = parser.parse_args()

    parse(args)