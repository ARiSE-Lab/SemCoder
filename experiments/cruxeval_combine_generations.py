# Reference: https://github.com/facebookresearch/cruxeval/blob/main/inference/combine_generations.py

import json
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str)
    args = parser.parse_args()

    
    new_dir = args.gen_dir
    files = os.listdir(args.gen_dir)

    for mode in ["orig", "raw"]:
        if mode == "orig":
            combined_json = {}
            current_keys = set()
            count = 0
            for input_json in files:
                if not input_json.endswith(".json"):
                    continue
                if input_json == "generations.json" or "raw" in input_json:
                    continue
                
                count += 1
                with open(os.path.join(args.gen_dir, input_json), "r") as fp:
                    input_json = json.load(fp)
                    input_json = {f"sample_{k}": v for k, v in input_json.items()}
                    keys = set(input_json.keys())
                    if keys.intersection(current_keys):
                        raise ValueError("Keys overlap")
                    combined_json.update(input_json)

            ## sort on keys and remove keys
            print(args.gen_dir, f"{count} files", len(combined_json))
            assert len(combined_json) == 800

            try: os.makedirs(new_dir)
            except: pass

            output_json = "generations.json"
            with open(os.path.join(new_dir, output_json), "w") as fp:
                json.dump(combined_json, indent=4, fp=fp)
        else:
            combined_json = {}
            current_keys = set()
            count = 0
            for input_json in files:
                if input_json == "generations_raw.json" or "raw" not in input_json:
                    continue
                if not input_json.endswith(".json"):
                    continue
                count += 1
                with open(os.path.join(args.gen_dir, input_json), "r") as fp:
                    input_json = json.load(fp)
                    input_json = {f"sample_{k}": v for k, v in input_json.items()}
                    keys = set(input_json.keys())
                    if keys.intersection(current_keys):
                        raise ValueError("Keys overlap")
                    combined_json.update(input_json)
            print(args.gen_dir, f"{count} files", len(combined_json))
            assert len(combined_json) == 800

            output_json = "generations_raw.json"
            with open(os.path.join(args.gen_dir, output_json), "w") as fp:
                json.dump(combined_json, indent=4, fp=fp)