import matplotlib.pyplot as plt
import elk
import torch as t
import os
import pandas as pd
import numpy as np
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer
from elk.extraction.prompt_dataset import PromptDataset, PromptConfig
from elk.extraction.extraction import ExtractionConfig, extract_hiddens, extract
import yaml

RPATH = "/fsx/home-rudolf/elk-reporters"
DEVICE = "cuda"


def load_config(name, rpath=RPATH):
    """Loads yaml config from RPATH/[name]/cfg.yaml"""
    with open(os.path.join(rpath, name, "cfg.yaml"), "r") as f:
        return yaml.safe_load(f)

def get_relevant_runs(reqs, rpath=RPATH):
    """Requirements is a dictionary of the form:
    property specification, e.g. "data.prompts.dataset"
    --->
    required value. For example:
    {
        "data.model": model_name
        "data.prompts.dataset": dataset_name,
        "net.net_name": "eigen",
    }
    """
    valid_reporters = []
    for name in os.listdir(rpath):
        cfg = load_config(name, rpath)
        add = True
        for prop, reqval in reqs.items():
            prop_parts = prop.split(".")
            temp = cfg
            for next_level in prop_parts:
                if next_level in temp.keys():
                    temp = temp[next_level]
                else:
                    add = False
                    break
            if temp != reqval:
                add = False
                break
        if add:
            valid_reporters.append(name)
    return valid_reporters

def get_eval(run_name):
    return pd.read_csv(os.path.join(RPATH, run_name, "eval.csv"))

def graph_eval(run_name):
    eval = get_eval(run_name)
    fig, ax = plt.subplots()
    ax.scatter(eval["layer"], eval["acc"], label="acc")
    ax.scatter(eval["layer"], eval["lr_acc"], label="LR acc")
    ax.legend()
    ax.set_title(f"{load_config(run_name)['data']['model']} on {load_config(run_name)['data']['prompts']['dataset']}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    fig.show()

def num_from_str(s):
    """Strip all non-numeric characters from s and interpret as a number"""
    num_str = ''
    for char in s:
        if char.isdigit():
            num_str += char
    if num_str == '':
        return None
    else:
        return int(num_str)

def get_reporters(run_name):
    r_dict = {
        layer_number : t.load(os.path.join(RPATH, run_name, "reporters", layer_name)).to(DEVICE)
        for (layer_number, layer_name) in [
            (num_from_str(layer_name), layer_name)
            for layer_name in os.listdir(os.path.join(RPATH, run_name, "reporters"))
            ]
        }
    return [r_dict[i] for i in range(len(r_dict))]

def dsget(dataset, i):
    dsi = dataset[i][0]
    true, false = list(dsi.to_string(answer_idx) for answer_idx in range(2))
    if dsi.label == 1:
        return (false, true)
    return (true, false)

def dsgets(dataset, num):
    return [dsget(dataset, i) for i in range(num)]

def model_hidden_states(
    model, 
    tokenizer,
    sentence,
    layer="all",
    tokens="all"
):
    inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)
    out_w_states = model(inputs["input_ids"], output_hidden_states=True)[1]
    # has shape [layers, tokens, hidden_size]
    if layer != "all":
        out_w_states = out_w_states[layer]
    if tokens != "all":
        if tokens == "last":
            out_w_states = out_w_states[:, -1]
        else:
            raise Exception(f"Invalid value for tokens: {tokens}")
    return out_w_states

def reporter_outputs(model, tokenizer, reporters, sentence):
    out_w_states = model_hidden_states(model, tokenizer, sentence)
    # the +1 in i+1 is because the first hidden state is the input embedding
    return t.stack(
        [
            reporter(out_w_states[i+1])[0]
            for i, reporter in enumerate(reporters)
        ],
        dim=0
    )

def reporter_output(model, tokenizer, reporter, sentence):
    return reporter_outputs(model, tokenizer, [reporter], sentence)[0]

def plot_2d_tensor_as_heatmap(t2d):
    fig, ax = plt.subplots()
    heatmap = ax.imshow(t2d.cpu().detach().numpy(), cmap="hot", interpolation="nearest")
    ax.set_xlabel("Token")
    ax.set_ylabel("Reporter (=layer)")
    plt.colorbar(heatmap)

def best_layer_num(run_name):
    layer_eval = get_eval(run_name)
    # find the row in the pandas dataframe layer_eval with the highest accuracy:
    best_layer = layer_eval["layer"][layer_eval["acc"].idxmax()]
    return best_layer

def best_reporter(run_name):
    best_layer = best_layer_num(run_name)
    reporter = get_reporters(run_name)[best_layer]
    return reporter

def best_layer_output(model, tokenizer, run_name, sentence):
    reporter = best_reporter(run_name)
    return reporter_outputs(model, tokenizer, [reporter], sentence)[0]

def reporter_predictions(
    reporter, model, tokenizer, contrast_pairs, layer, tokens="last"
):
    """Assumes the true statements come first in contrast_pairs"""
    predictions = []
    for pair in contrast_pairs:
        true_str, false_str = pair
        
        # true_tokens = tokenizer(true_str, return_tensors="pt").to(DEVICE)["input_ids"]
        # false_tokens = tokenizer(false_str, return_tensors="pt").to(DEVICE)["input_ids"]
        
        state_true = model_hidden_states(model, tokenizer, true_str, layer, tokens)
        state_false = model_hidden_states(model, tokenizer, false_str, layer, tokens)
        
        predicted = reporter.predict(state_true, state_false)
        predictions.append(predicted.detach()[0].item())
    return t.tensor(predictions)
