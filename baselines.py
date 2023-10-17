import argparse
import copy
import json
import random
from tqdm import tqdm
import numpy as np
import torch
import math
from torch.utils.data.dataloader import DataLoader
from transformers import XLMRobertaTokenizer, RobertaForQuestionAnswering
from utils.sam import SAM

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


def qa_loss(start_logits, end_logits, start_positions, end_positions, return_start_end=False):
    ignored_index = start_logits.size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index, reduce=False)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss) / 2

    if not return_start_end:
        return total_loss
    else:
        return total_loss, start_loss, end_loss


def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def reset(model, model_state):
    model.load_state_dict(model_state)


def reset_model_and_optimizer(model, model_state, optimizer, optimizer_state):
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)


def get_args():
    parser = argparse.ArgumentParser()
    # TTA args
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--method", type=str, default="Forward", choices=["Forward", "Tent", "EATA", "OIL", "SAR"])
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


def Forward(args, model, dataset, examples, features, tokenizer, language):
    eval_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
    model.eval()
    all_results = []
    for batch in tqdm(eval_loader):
        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            example_indices = batch[3]
            outputs = model(**inputs)
        start_logits, end_logits = to_list(outputs[0]), to_list(outputs[1])
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = SquadResult(unique_id, start_logits[i], end_logits[i])
            all_results.append(result)
    predictions = compute_predictions_logits(examples, features, all_results, args.n_best_size, args.max_answer_length,
                                             False, None, None, None, False, False, 0, tokenizer,
                                             map_to_origin=(language != "zh"))
    return predictions


def Tent(args, model, dataset, examples, features, tokenizer, language):
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
        outputs = model(**inputs)
        start_loss = entropy(outputs[0]).mean()
        end_loss = entropy(outputs[1]).mean()
        loss = (start_loss + end_loss) / 2
        loss.backward()
        optimizer.step()
        model.zero_grad()
        start_logits, end_logits = to_list(outputs[0]), to_list(outputs[1])
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = SquadResult(unique_id, start_logits[i], end_logits[i])
            all_results.append(result)
    predictions = compute_predictions_logits(examples, features, all_results, args.n_best_size, args.max_answer_length,
                                             False, None, None, None, False, False, 0, tokenizer,
                                             map_to_origin=(language != "zh"))
    return predictions


def EATA(args, model, dataset, examples, features, tokenizer, language, fishers):
    e_margin = 0.4 * math.log(512)
    fisher_alpha = 1 / 2000

    def post_process_entropy(tensor):
        selected_idx = torch.where(tensor < e_margin)
        tensor = tensor[selected_idx]
        coeff = 1 / (torch.exp(tensor.clone().detach() - e_margin))
        tensor = tensor.mul(coeff)
        loss = tensor.mean()
        ewc_loss = 0
        for name, param in model.named_parameters():
            if name in fishers:
                ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1]) ** 2).sum()
        loss += ewc_loss
        return loss

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
        outputs = model(**inputs)
        start_entropy = entropy(outputs[0])
        end_entropy = entropy(outputs[1])
        all_loss = (post_process_entropy(start_entropy) + post_process_entropy(end_entropy)) / 2
        all_loss.backward()
        optimizer.step()
        model.zero_grad()
        start_logits, end_logits = to_list(outputs[0]), to_list(outputs[1])
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = SquadResult(unique_id, start_logits[i], end_logits[i])
            all_results.append(result)
    predictions = compute_predictions_logits(examples, features, all_results, args.n_best_size, args.max_answer_length,
                                             False, None, None, None, False, False, 0, tokenizer,
                                             map_to_origin=(language != "zh"))
    return predictions


def OIL(args, model, dataset, examples, features, tokenizer, language):
    eval_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    teacher_model = copy.deepcopy(model)
    teacher_model.eval()
    all_results = []
    for batch in tqdm(eval_loader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
        }
        example_indices = batch[3]
        with torch.no_grad():
            outputs = model(**inputs)
            source_outputs = teacher_model(**inputs)
            pseudo_start_logits = source_outputs[0]
            pseudo_end_logits = source_outputs[1]
            beta = 1
            start_logits = 2 * outputs[0] - pseudo_start_logits - beta * (outputs[0] - pseudo_start_logits)
            end_logits = 2 * outputs[1] - pseudo_end_logits - beta * (outputs[1] - pseudo_end_logits)
            outputs = (start_logits, end_logits)
            start_logits, end_logits = to_list(outputs[0]), to_list(outputs[1])

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = SquadResult(unique_id, start_logits[i], end_logits[i])
            all_results.append(result)

        model.train()
        with torch.no_grad():
            source_outputs = teacher_model(**inputs)
            pseudo_start_logits = source_outputs[0].detach()
            pseudo_end_logits = source_outputs[1].detach()
            pseudo_start_positions = torch.argmax(pseudo_start_logits, dim=1)
            pseudo_end_positions = torch.argmax(pseudo_end_logits, dim=1)
        adapted_outputs = model(**inputs)
        start_logits = adapted_outputs[0]
        end_logits = adapted_outputs[1]

        _, loss_hard_start, loss_hard_end = qa_loss(start_logits, end_logits,
                                                    copy.deepcopy(pseudo_start_positions),
                                                    copy.deepcopy(pseudo_end_positions),
                                                    return_start_end=True)  # (batch,)
        start_logits_2 = (start_logits + start_logits - pseudo_start_logits)
        end_logits_2 = (end_logits + end_logits - pseudo_end_logits)
        _, loss_clean_start, loss_clean_end = qa_loss(start_logits_2, end_logits_2,
                                                      copy.deepcopy(pseudo_start_positions),
                                                      copy.deepcopy(pseudo_end_positions),
                                                      return_start_end=True)
        echo = args.topk
        loss = (torch.sum((loss_hard_start < echo) * loss_clean_start) / (
                torch.sum((loss_hard_start < echo)) + 1e-10) +
                torch.sum((loss_hard_end < echo) * loss_clean_end) / (
                        torch.sum((loss_hard_end < echo)) + 1e-10)) / 2.

        loss.backward()
        optimizer.step()
        model.zero_grad()

        for param1, param2 in zip(teacher_model.parameters(), model.parameters()):
            param1.data = 0.99 * param1.data + (1 - 0.99) * param2.data
    predictions = compute_predictions_logits(examples, features, all_results, args.n_best_size, args.max_answer_length,
                                             False, None, None, None, False, False, 0, tokenizer,
                                             map_to_origin=(language != "zh"))
    return predictions


def SAR(args, model, dataset, examples, features, tokenizer, language, model_state):
    e_margin = 0.4 * math.log(512)
    ema = None
    eval_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, momentum=0.9)
    model.eval()
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    all_results = []
    for batch in tqdm(eval_loader):
        optimizer.zero_grad()
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
        }
        example_indices = batch[3]
        outputs = model(**inputs)
        start_entropy = entropy(outputs[0])
        end_entropy = entropy(outputs[1])
        selected_idx_start = torch.where(start_entropy < e_margin)
        selected_idx_end = torch.where(end_entropy < e_margin)
        start_entropy = start_entropy[selected_idx_start]
        end_entropy = end_entropy[selected_idx_end]
        all_loss = (start_entropy.mean() + end_entropy.mean()) / 2
        all_loss.backward()
        optimizer.first_step(zero_grad=True)

        outputs2 = model(**inputs)
        start_entropy2 = entropy(outputs2[0])
        end_entropy2 = entropy(outputs2[1])
        start_entropy2 = start_entropy2[selected_idx_start]
        end_entropy2 = end_entropy2[selected_idx_end]

        selected_idx_start2 = torch.where(start_entropy2 < e_margin)
        selected_idx_end2 = torch.where(end_entropy2 < e_margin)

        start_entropy2 = start_entropy2[selected_idx_start2]
        end_entropy2 = end_entropy2[selected_idx_end2]

        loss_second = (start_entropy2.mean() + end_entropy2.mean()) / 2
        if not np.isnan(loss_second.item()):
            ema = update_ema(ema, loss_second.item())

        loss_second.backward()
        optimizer.second_step(zero_grad=True)

        reset_flag = False
        if ema is not None:
            if ema < 0.2:
                print("ema < 0.2, now reset the model")
                reset_flag = True
        if reset_flag:
            reset_model_and_optimizer(model, model_state, optimizer, optimizer_state)
            ema = None
        start_logits, end_logits = to_list(outputs[0]), to_list(outputs[1])
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = SquadResult(unique_id, start_logits[i], end_logits[i])
            all_results.append(result)
    predictions = compute_predictions_logits(examples, features, all_results, args.n_best_size, args.max_answer_length,
                                             False, None, None, None, False, False, 0, tokenizer,
                                             map_to_origin=(language != "zh"))
    return predictions


def compute_fisher(args, model, dataset):
    fishers = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fishers[name] = [0, param.data.clone().detach()]
    eval_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.eval()
    print("compute fishers...")
    for batch in tqdm(eval_loader):
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }
        loss = model(**inputs)[0]
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                fishers.update({name: [fisher, param.data.clone().detach()]})
        optimizer.zero_grad()
    for name, param in model.named_parameters():
        if param.requires_grad:
            fishers[name][0] /= len(eval_loader)
    print("fishers computed!")
    return fishers


def main():
    args = get_args()
    set_seed(args.seed)
    dataset_paths = json.load(open("utils/dataset_paths.json", "r"))
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_path)
    model = RobertaForQuestionAnswering.from_pretrained(args.model_path).to(args.device)
    for name, param in model.named_parameters():
        if "LayerNorm" not in name:
            param.requires_grad = False
    if args.method == "EATA":
        warmup_dataset, _, _ = load_and_cache_examples(args, "download/warmup.json", tokenizer, True)
        fishers = compute_fisher(args, model, warmup_dataset)
    else:
        fishers = None
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
            if args.method == "Forward":
                predictions = Forward(args, model, dataset, examples, features, tokenizer, subset_path[-7:-5])
            elif args.method == "Tent":
                predictions = Tent(args, model, dataset, examples, features, tokenizer, subset_path[-7:-5])
            elif args.method == "EATA":
                predictions = EATA(args, model, dataset, examples, features, tokenizer, subset_path[-7:-5], fishers)
            elif args.method == "OIL":
                if "noiseQA" or "XQuAD" in dataset_path:
                    args.topk = 0.5
                else:
                    args.topk = float('inf')
                predictions = OIL(args, model, dataset, examples, features, tokenizer, subset_path[-7:-5])
            elif args.method == "SAR":
                predictions = SAR(args, model, dataset, examples, features, tokenizer, subset_path[-7:-5], model_state)
            else:
                predictions = None
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
