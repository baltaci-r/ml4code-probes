import json
import os
from pathlib import Path
import argparse

from models.base import InterpretableModel
from tasks.global_feature_pred import GlobalFeaturePredTask
from tasks.token_label_pred import TokenLabelPredTask
from tasks.numeracy import NumeracyTask



def get_task(model, task):

    return {
        'global_feature': GlobalFeaturePredTask,
        'token_label': TokenLabelPredTask,
        'numeracy': NumeracyTask,
    }[task](model)


def main(args):
    pretrained = InterpretableModel(args.model_name)
    task = get_task(pretrained, args.task)
    data = task.load_data(args.path, args.results_path, args.data_path, args.configs.split(','))
    # if args.run:  # WHY ?
    results = task.run(data, args.num, args.results_path)
    with open(os.path.join(args.results_path, 'all_results.json'), "w") as f:
        json.dump(results, f, indent=4)


def update_args(args):
    cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    data_dir = cur_dir / f'../../probing_data/{args.task}/{args.model_name}'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    args.path = cur_dir / f'../../probing_data/{args.task}/{args.model_name}/'  # prepared_data_test.pt'
    # args.results_path = cur_dir / f"../../results/{args.task}_{args.model_name.replace('/', '_')}_{args.num}.json"
    args.results_path = cur_dir / f"../../results/{args.task}_{args.model_name.replace('/', '_')}"
    if not os.path.isdir(args.results_path):
        os.makedirs(args.results_path )
    return args


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="microsoft/codebert-base")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--task', type=str, choices=['global_feature', 'token_label', 'numeracy'])
    parser.add_argument('--num', type=int, help='Number of probing samples')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--configs', type=str, required=True)
    args = parser.parse_args()
    args = update_args(args)

    main(args)