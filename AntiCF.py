import argparse
import copy
import json
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import XLMRobertaTokenizer
from utils.roberta_with_sideblock import RobertaForQuestionAnswering

from utils.metrics import compute_predictions_logits
from utils.mlqa_evaluate import evaluate_mlqa
from utils.process import SquadResult, load_and_cache_examples
from utils.squad_evaluate import evaluate_squad


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def kl(source, target):
    return torch.sum(target.softmax(dim=-1) * (target.log_softmax(dim=-1) - source.log_softmax(dim=-1)), dim=-1)


def entropy(tensor):
    return -(tensor.softmax(dim=-1) * tensor.log_softmax(dim=-1)).sum(dim=-1)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def reset(model, model_state):
    model.load_state_dict(model_state)


def get_args():
    parser = argparse.ArgumentParser()
    # TTA args
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--device", type=str)
    parser.add_argument("--model_path", type=str)

    # QA args
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--n_best_size", type=int, default=20)
    parser.add_argument("--max_answer_length", type=int, default=30)
    parser.add_argument("--doc_stride", type=int, default=128)

    # save args
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    return args


def AntiCF(args, model, dataset, examples, features, tokenizer, language):
    eval_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.eval()
    all_results = []
    for batch in tqdm(eval_loader):
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
        }
        example_indices = batch[3]
        source_outputs, adapter_outputs = model(**inputs)
        start_loss = entropy(adapter_outputs[0]).mean()
        end_loss = entropy(adapter_outputs[1]).mean()
        start_kl = kl(adapter_outputs[0], source_outputs[0]).mean()
        end_kl = kl(adapter_outputs[1], source_outputs[1]).mean()
        loss = (start_loss + end_loss) * (1 - args.alpha) + (start_kl + end_kl) * args.alpha / 2
        loss.backward()
        optimizer.step()
        model.zero_grad()

        start_logits, end_logits = to_list(adapter_outputs[0]), to_list(adapter_outputs[1])
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = SquadResult(unique_id, start_logits[i], end_logits[i])
            all_results.append(result)
    predictions = compute_predictions_logits(examples, features, all_results, args.n_best_size, args.max_answer_length,
                                             False, None, None, None, False, False, 0, tokenizer,
                                             map_to_origin=(language != "zh"))
    return predictions


def warmup(args, model, dataset):
    warmup_loader = list(DataLoader(dataset, shuffle=True, batch_size=8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    model.train()
    step = 0
    print("warmup start...")
    for batch in tqdm(warmup_loader):
        step += 1
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }
        loss = model(**inputs)
        loss.backward()
        if step % 4 == 0:
            step = 0
            optimizer.step()
            model.zero_grad()
    print("warmup finished!")


def main():
    args = get_args()
    set_seed(args.seed)
    dataset_paths = json.load(open("utils/dataset_paths.json", "r"))
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_path)
    model = RobertaForQuestionAnswering.from_pretrained(args.model_path).to(args.device)
    cls_dict = copy.deepcopy(model.qa_outputs.state_dict())
    model.adapter_outputs.load_state_dict(cls_dict)
    for name, param in model.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
    warmup_dataset, _, _ = load_and_cache_examples(args, "download/warmup.json", tokenizer, True)
    warmup(args, model, warmup_dataset)
    model_state = copy.deepcopy(model.state_dict())
    all_results = {"avg": {}}
    all_em, all_f1 = [], []

    for dataset_path in dataset_paths:
        all_results[dataset_path] = {}
        dataset_em, dataset_f1 = [], []
        for subset_path in dataset_paths[dataset_path]:
            print("Inferencing {} from {}".format(subset_path, dataset_path))
            dataset, examples, features = load_and_cache_examples(args, dataset_path + "/" + subset_path, tokenizer,
                                                                  False)
            predictions = AntiCF(args, model, dataset, examples, features, tokenizer, subset_path[-7:-5])
            reset(model, model_state)
            with open(dataset_path + "/" + subset_path, encoding="utf-8", mode="r") as f:
                data = json.load(f)["data"]
            if "MLQA" in dataset_path:
                result = evaluate_mlqa(data, predictions, subset_path[-7:-5])
            else:
                result = evaluate_squad(data, predictions)
            em, f1 = result["exact_match"], result["f1"]
            print("dataset:{}    subset:{}    em:{:.4f}    f1:{:.4f}".format(dataset_path, subset_path, em, f1))
            dataset_em.append(em)
            dataset_f1.append(f1)
        dataset_em.append(sum(dataset_em) / len(dataset_em))
        dataset_f1.append(sum(dataset_f1) / len(dataset_f1))
        all_results[dataset_path]["em"], all_results[dataset_path]["f1"] = dataset_em, dataset_f1
        all_em.append(sum(dataset_em) / len(dataset_em))
        all_f1.append(sum(dataset_f1) / len(dataset_f1))
    all_em.append(sum(all_em) / len(all_em))
    all_f1.append(sum(all_f1) / len(all_f1))
    all_results["avg"]["em"], all_results["avg"]["f1"] = all_em, all_f1
    print("Finish all inference, final results:")
    print(all_results)
    if args.output_path is not None:
        out = open(args.output_path, mode="w", encoding="utf-8")
        json.dump(all_results, out, indent=1)
        out.close()


if __name__ == "__main__":
    main()
