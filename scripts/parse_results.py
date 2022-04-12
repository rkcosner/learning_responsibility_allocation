import json
import argparse
import numpy as np

def parse(args):
    rjson = json.load(open(args.results_file, "r"))
    for k in rjson:
        print("{} = {}".format(k, np.mean(rjson[k])))
    print("num_scenes: {}".format(len(rjson["scene_index"])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
        help="A json file containing evaluation results"
    )

    args = parser.parse_args()

    parse(args)