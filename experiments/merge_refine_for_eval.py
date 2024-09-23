import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_pred_file', type=str, required=True)
    parser.add_argument('--refine_pred_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    # replace the original prediction with the refined prediction
    with open(args.orig_pred_file, 'r') as f:
        orig_preds = f.readlines()
    orig_preds = [json.loads(p) for p in orig_preds]
    orig_key_to_pred = {p["task_id"]: p for p in orig_preds}

    with open(args.refine_pred_file, 'r') as f:
        refine_preds = f.readlines()
    refine_preds = [json.loads(p) for p in refine_preds]
    refine_key_to_pred = {p["task_id"]: p for p in refine_preds}

    for k in orig_key_to_pred:
        if k in refine_key_to_pred:
            orig_key_to_pred[k] = refine_key_to_pred[k]

    with open(args.output_file, 'w') as f:
        for k in orig_key_to_pred:
            f.write(json.dumps(orig_key_to_pred[k]) + '\n')
            

    