import json
import argparse
import numpy as np
import os
from pprint import pprint



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
        results["all_off_road_rate_rate"] * 100
    ]

    results_str = ["{:.3f}".format(r) for r in results_str]

    print(",".join(results_str))


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