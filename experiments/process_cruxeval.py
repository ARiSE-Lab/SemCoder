"""
Parse the raw generations of the models to extract the assertions
"""

import argparse
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str)
    parser.add_argument("--task", type=str, choices=['i', 'o'])
    args = parser.parse_args()

    raw_gen_file = os.path.join(args.gen_dir, "generations_raw.json")
    with open(raw_gen_file, "r") as fp:
        raw_gen = json.load(fp)

    processed_gen = {}
    for k in raw_gen:
        processed_gen[k] = []
        for raw in raw_gen[k]:
            # raw = raw_gen[k][0] 
            # look for content within the SECOND occurence of [ANSWER] and [/ANSWER]
            # first the last occurence of [ANSWER]
            if "\nANSWER]" in raw:
                start = raw.rfind("ANSWER]") + len("ANSWER]")
            else:
                start = raw.rfind("[ANSWER]") + len("[ANSWER]")
            end = raw.rfind("[/ANS")
            if start == -1:
                start = 0
            if end == -1 or end < start:
                end = len(raw)
            answer = raw[start:end]
            if args.task == 'o':
                if "==" in answer:
                    answer = answer.split("==")[1].strip()
                if "assert" in answer:
                    answer = answer.split("assert")[1].strip()
            elif args.task == 'i':
                if "==" in answer:
                    answer = answer.split("==")[0].strip()
                if "assert" in answer:
                    answer = answer.split("assert")[1].strip()
            answer = answer.replace('"', "'") # replace double quotes with single quotes
            processed_gen[k].append(answer)

    with open(os.path.join(args.gen_dir, "generations.json"), "w") as fp:
        print("Writing to", os.path.join(args.gen_dir, "generations.json"))
        json.dump(processed_gen, indent=4, fp=fp)