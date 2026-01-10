import argparse
import json
import textwrap
from argparse import Namespace
from pathlib import Path

import numpy as np

from tempora.common.utils import print_arguments, save_results


# Calculate continuous utility metrics from offline evaluation logs. Given a baseline prediction time (ρ) and a HCI
# threshold (T), the experienced delay is sum of the post-hoc (extrinsic) adaptation overhead from the previous batch
# and the intrinsic adaptation overhead of the current batch. The hyperbolic decay constant is k = 1 / (T - ρ). 
def project_continuous(data, response_budget, decay_threshold):
    if decay_threshold - response_budget <= 0:
        raise ValueError("No overhead budget available, adaptation cannot improve under this constraint.")
    
    batch_accuracy = []
    batch_discount = []

    decay_constant = 1.0 / (decay_threshold - response_budget)
    carry_overhead = 0.0
    for i, batch in enumerate(data["batch_summaries"]):
        overhead = max(0.0, carry_overhead + batch["prediction_time"] - response_budget)
        discount = 1.0 / (1.0 + decay_constant * overhead)

        data["batch_summaries"][i]["overhead"] = overhead
        data["batch_summaries"][i]["discount"] = discount

        batch_accuracy.append(batch["num_correct"] / batch["num_samples"] if batch["num_samples"] > 0 else 0.0)
        batch_discount.append(discount)

        carry_overhead = batch["adaptation_time"]

    batch_accuracy = np.array(batch_accuracy)
    batch_discount = np.array(batch_discount)

    timeliness = np.mean(batch_discount)
    alignment = np.cov(batch_accuracy, batch_discount)[0, 1]
    accuracy = np.mean(batch_accuracy)
    utility = accuracy * timeliness + alignment

    data["timeliness"] = timeliness
    data["alignment"] = alignment
    data["accuracy"] = accuracy
    data["utility"] = utility

    output = textwrap.dedent(f"""\
        {"-" * 32}
        Timeliness : {timeliness:.2%}
        Alignment  : {alignment:+.4f} 
        Accuracy   : {accuracy:.2%}
        Utility    : {utility:.2%}
        {"-" * 32}""")
    
    print(output)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)  # File or directory containing source JSON evaluation logs
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--response-budget", type=float, default=50.0)
    parser.add_argument("--decay-threshold", type=float, default=100.0)

    args = parser.parse_args()
    print_arguments(args)

    files = []
    if args.input.is_dir():
        for file in sorted(args.input.glob("*.json")):
            files.append(file)
    elif args.input.is_file():
        files = [args.input]
    else:
        raise ValueError(f"Input doesn't exist: {args.input}")

    for file in files:
        print(f"Processing {file}")
        with open(file) as f:
            data = json.load(f)

        rs = {}
        for d, r in data["results"].items():
            rs[d] = project_continuous(r, args.response_budget, args.decay_threshold)

        data["arguments"]["output_dir"] = args.output_dir
        data["arguments"]["source_filepath"] = file
        data["arguments"]["decay_threshold"] = args.decay_threshold
        data["arguments"]["response_budget"] = args.response_budget
        save_results(rs, Namespace(**data["arguments"]))
