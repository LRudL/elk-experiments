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


from elk.extraction.prompt_dataset import Prompt, PromptDataset, PromptConfig
from elk.utils import (
    assert_type,
    float32_to_int16,
    infer_label_column,
    select_train_val_splits,
    select_usable_devices,
)
from elk.extraction.generator import _GeneratorBuilder
from dataclasses import dataclass, InitVar
from datasets import (
    Array3D,
    DatasetDict,
    Features,
    get_dataset_config_info,
    Sequence,
    Split,
    SplitDict,
    SplitInfo,
    Value,
)
from simple_parsing.helpers import field, Serializable
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
)
from typing import Iterable, Literal, Union, List
import logging
import torch

RPATH = "/fsx/home-rudolf/elk-reporters"
DEVICE = "cuda:1"


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
        
        state_true = model_hidden_states(model, tokenizer, true_str, layer, tokens)
        state_false = model_hidden_states(model, tokenizer, false_str, layer, tokens)
        
        predicted = reporter.predict(state_true, state_false)
        predictions.append(predicted.detach()[0].item())
    return t.tensor(predictions)























# copy of elk.extraction.extract_hiddens
@torch.no_grad()
def extract_hiddens_gen(
    # cfg: ExtractionConfig,
    prompt_ds,
    model,
    tokenizer, # MAKE SURE I THAS truncation_side="left"
    layers = None, # layer_indices
    token_loc = "last",
    rank = 0,
    world_size = 1,
    device = DEVICE
):
    # prompt_ds = PromptDataset(cfg.prompts, rank, world_size, split)
    num_variants = 1
    if rank == 0:
        prompt_names = prompt_ds.prompter.all_template_names
        if num_variants >= 1:
            print(
                f"Using {num_variants} prompts per example: {prompt_names}"
            )
        elif num_variants == -1:
            print(f"Using all prompts per example: {prompt_names}")
        else:
            raise ValueError(f"Invalid prompt num_variants: {num_variants}")

    # AutoModel should do the right thing here in nearly all cases. We don't actually
    # care what head the model has, since we are just extracting hidden states.
    # model = AutoModel.from_pretrained(cfg.model, torch_dtype="auto").to(device)
    # TODO: Maybe also make this configurable?
    # We want to make sure the answer is never truncated
    # tokenizer = AutoTokenizer.from_pretrained(cfg.model, truncation_side="left")

    # TODO: test whether using sep_token is important, but this seems low priority
    # sep_token = tokenizer.sep_token or "\n"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    is_enc_dec = model.config.is_encoder_decoder

    def tokenize(prompt: Prompt, idx: int, **kwargs):
        return tokenizer(
            ([prompt.to_string(idx)]),
            padding=True,
            return_tensors="pt",
            truncation=True,
            **kwargs,
        ).to(device)

    # This function returns the flattened questions and answers. After inference we
    # need to reshape the results.
    def collate(prompts: list[Prompt]) -> list[list[BatchEncoding]]:
        return [[tokenize(prompt, i) for i in range(2)] for prompt in prompts]

    # If this is an encoder-decoder model we don't need to run the decoder at all.
    # Just strip it off, making the problem
    # equivalent to a regular encoder-only model.
    if is_enc_dec:
        # This isn't actually *guaranteed* by HF, but it's true for all existing models
        if not hasattr(model, "get_encoder") or not callable(model.get_encoder):
            raise ValueError(
                "Encoder-decoder model doesn't have expected get_encoder() method"
            )

        model = assert_type(PreTrainedModel, model.get_encoder())

    # Iterating over questions
    layer_indices = layers or tuple(range(model.config.num_hidden_layers))
    for prompts in prompt_ds:
        inputs = collate(prompts)
        hidden_dict = {
            f"hidden_{layer_idx}": torch.empty(
                prompt_ds.num_variants,
                2,  # contrast pair
                model.config.hidden_size,
                device=device,
                dtype=torch.float32,
            )
            for layer_idx in layer_indices
        }
        variant_ids = [prompt.template.name for prompt in prompts]
        # decode so that we know exactly what the input was
        text_inputs = [
            [
                tokenizer.decode(
                    assert_type(torch.Tensor, variant_inputs[0].input_ids)[0]
                ),
                tokenizer.decode(
                    assert_type(torch.Tensor, variant_inputs[1].input_ids)[0]
                ),
            ]
            for variant_inputs in inputs
        ]

        # Iterate over variants
        for i, variant_inputs in enumerate(inputs):
            # Iterate over answers
            for j, inpt in enumerate(variant_inputs):
                outputs = model(**inpt, output_hidden_states=True)

                hiddens = (
                    outputs.get("decoder_hidden_states") or outputs["hidden_states"]
                )
                # First element of list is the input embeddings
                hiddens = hiddens[1:]

                # Throw out layers we don't care about
                hiddens = [hiddens[i] for i in layer_indices]

                # Current shape of each element: (batch_size, seq_len, hidden_size)
                if token_loc == "first":
                    hiddens = [h[..., 0, :] for h in hiddens]
                elif token_loc == "last":
                    hiddens = [h[..., -1, :] for h in hiddens]
                elif token_loc == "mean":
                    hiddens = [h.mean(dim=-2) for h in hiddens]
                else:
                    raise ValueError(f"Invalid token_loc: {token_loc}")

                for layer_idx, hidden in zip(layer_indices, hiddens):
                    hidden_dict[f"hidden_{layer_idx}"][i, j] = hidden

        assert all([prompts[0].label == prompt.label for prompt in prompts])
        yield dict(
            label=prompts[0].label,
            variant_ids=variant_ids,
            text_inputs=text_inputs,
            **hidden_dict,
        )





# copy of the above, not requiring a dataset
@torch.no_grad()
def extract_hidden_for_prompt(
    prompts,
    model,
    tokenizer, # MAKE SURE IT HAS truncation_side="left"
    layers = None, # layer_indices
    token_loc = "last",
    device = DEVICE
):
    num_variants = len(prompts)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    is_enc_dec = model.config.is_encoder_decoder

    def tokenize(prompt: Prompt, idx: int, **kwargs):
        return tokenizer(
            ([prompt.to_string(idx)]),
            padding=True,
            return_tensors="pt",
            truncation=True,
            **kwargs,
        ).to(device)

    # This function returns the flattened questions and answers. After inference we
    # need to reshape the results.
    def collate(prompts: list[Prompt]) -> list[list[BatchEncoding]]:
        return [[tokenize(prompt, i) for i in range(2)] for prompt in prompts]

    # If this is an encoder-decoder model we don't need to run the decoder at all.
    # Just strip it off, making the problem
    # equivalent to a regular encoder-only model.
    if is_enc_dec:
        # This isn't actually *guaranteed* by HF, but it's true for all existing models
        if not hasattr(model, "get_encoder") or not callable(model.get_encoder):
            raise ValueError(
                "Encoder-decoder model doesn't have expected get_encoder() method"
            )

        model = assert_type(PreTrainedModel, model.get_encoder())

    # Iterating over questions
    layer_indices = layers or tuple(range(model.config.num_hidden_layers))
    
    inputs = collate(prompts)
    hidden_dict = {
        f"hidden_{layer_idx}": torch.empty(
            num_variants,
            2,  # contrast pair
            model.config.hidden_size,
            device=device,
            dtype=torch.float32,
        )
        for layer_idx in layer_indices
    }
    variant_ids = [prompt.template.name for prompt in prompts]
    # decode so that we know exactly what the input was
    text_inputs = [
        [
            tokenizer.decode(
                assert_type(torch.Tensor, variant_inputs[0].input_ids)[0]
            ),
            tokenizer.decode(
                assert_type(torch.Tensor, variant_inputs[1].input_ids)[0]
            ),
        ]
        for variant_inputs in inputs
    ]

    # Iterate over variants
    for i, variant_inputs in enumerate(inputs):
        # Iterate over answers
        for j, inpt in enumerate(variant_inputs):
            outputs = model(**inpt, output_hidden_states=True)

            hiddens = (
                outputs.get("decoder_hidden_states") or outputs["hidden_states"]
            )
            # First element of list is the input embeddings
            hiddens = hiddens[1:]

            # Throw out layers we don't care about
            hiddens = [hiddens[i] for i in layer_indices]

            # Current shape of each element: (batch_size, seq_len, hidden_size)
            if token_loc == "first":
                hiddens = [h[..., 0, :] for h in hiddens]
            elif token_loc == "last":
                hiddens = [h[..., -1, :] for h in hiddens]
            elif token_loc == "mean":
                hiddens = [h.mean(dim=-2) for h in hiddens]
            else:
                raise ValueError(f"Invalid token_loc: {token_loc}")

            for layer_idx, hidden in zip(layer_indices, hiddens):
                hidden_dict[f"hidden_{layer_idx}"][i, j] = hidden

    assert all([prompts[0].label == prompt.label for prompt in prompts])
    return dict(
        label=prompts[0].label,
        variant_ids=variant_ids,
        text_inputs=text_inputs,
        **hidden_dict,
    )


def reporter_predictions2(
    reporter, model, tokenizer, dataset_examples, layer, tokens="last"
):
    """Assumes the true statements come first in contrast_pairs"""
    predictions = np.array([])
    for prompts in dataset_examples:
        
        hidden = extract_hidden_for_prompt(prompts, model, tokenizer, layers=[layer], token_loc=tokens)
        
        hidden_state = hidden[f"hidden_{layer}"]
        hidden_state0 = hidden_state[:,0]
        hidden_state1 = hidden_state[:,1]
        
        preds = reporter.predict(hidden_state0, hidden_state1)
        
        correct = prompts[0].label # must be same for all prompts
        
        if correct == 1:
            preds = preds * -1
        
        predictions = np.concatenate([predictions, preds.detach().cpu().numpy()])
    return predictions

def reporter_accuracy(
    reporter,
    model,
    tokenizer,
    dataset,
    layer,
    tokens="last",
    num_examples = 50
):
    dataset_examples = [
        dataset[i] for i in range(num_examples)
    ]
    predictions = reporter_predictions2(
        reporter, model, tokenizer, dataset_examples, layer, tokens
    )
    accuracy = (predictions > 0).sum() / predictions.shape[0]
    return accuracy
    
    
    
    
    
    
    
# copy of the above, not requiring a dataset
@torch.no_grad()
def extract_hidden_from_str(
    contrast_pair,
    model,
    tokenizer, # MAKE SURE IT HAS truncation_side="left"
    layers = None, # layer_indices
    token_loc = "last",
    device = DEVICE
):
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    is_enc_dec = model.config.is_encoder_decoder

    def tokenize(str, **kwargs):
        return tokenizer(
            ([str]),
            padding=True,
            return_tensors="pt",
            truncation=True,
            **kwargs,
        ).to(device)

    # This function returns the flattened questions and answers. After inference we
    # need to reshape the results.
    def collate(contrast_pair: list[str]) -> list[BatchEncoding]:
        return [tokenize(str) for str in contrast_pair]

    # If this is an encoder-decoder model we don't need to run the decoder at all.
    # Just strip it off, making the problem
    # equivalent to a regular encoder-only model.
    if is_enc_dec:
        # This isn't actually *guaranteed* by HF, but it's true for all existing models
        if not hasattr(model, "get_encoder") or not callable(model.get_encoder):
            raise ValueError(
                "Encoder-decoder model doesn't have expected get_encoder() method"
            )

        model = assert_type(PreTrainedModel, model.get_encoder())

    # Iterating over questions
    layer_indices = layers or tuple(range(model.config.num_hidden_layers))
    
    inputs = collate(contrast_pair)
    
    hidden_dict = {
        f"hidden_{layer_idx}": torch.empty(
            2,  # contrast pair
            model.config.hidden_size,
            device=device,
            dtype=torch.float32,
        )
        for layer_idx in layer_indices
    }
    # variant_ids = [prompt.template.name for prompt in prompts]
    # decode so that we know exactly what the input was
    # text_inputs = [
    #     [
    #         tokenizer.decode(
    #             assert_type(torch.Tensor, variant_inputs[0].input_ids)[0]
    #         ),
    #         tokenizer.decode(
    #             assert_type(torch.Tensor, variant_inputs[1].input_ids)[0]
    #         ),
    #     ]
    #     for variant_inputs in inputs
    # ]

    # Iterate over variants
    for j, inpt in enumerate(inputs):
        outputs = model(**inpt, output_hidden_states=True)

        hiddens = (
            outputs.get("decoder_hidden_states") or outputs["hidden_states"]
        )
        # First element of list is the input embeddings
        hiddens = hiddens[1:]

        # Throw out layers we don't care about
        hiddens = [hiddens[i] for i in layer_indices]

        # Current shape of each element: (batch_size, seq_len, hidden_size)
        if token_loc == "first":
            hiddens = [h[..., 0, :] for h in hiddens]
        elif token_loc == "last":
            hiddens = [h[..., -1, :] for h in hiddens]
        elif token_loc == "mean":
            hiddens = [h.mean(dim=-2) for h in hiddens]
        else:
            raise ValueError(f"Invalid token_loc: {token_loc}")

        for layer_idx, hidden in zip(layer_indices, hiddens):
            hidden_dict[f"hidden_{layer_idx}"][j] = hidden

    return dict(
        # label=prompts[0].label,
        # variant_ids=variant_ids,
        # text_inputs=text_inputs,
        **hidden_dict,
    )

def reporter_predictions3(
    reporter, model, tokenizer, contrast_pairs, layer, tokens="last"
):
    """Assumes the true statements come first in contrast_pairs"""
    predictions = np.array([])
    for contrast_pair in contrast_pairs:
        not_pair = False
        if type(contrast_pair) != tuple:
            # ugly hacks ...
            # (just need something to match type of extract_hidden_from_str)
            contrast_pair = [contrast_pair, contrast_pair]
            not_pair = True
        hidden = extract_hidden_from_str(
            contrast_pair, model, tokenizer, layers=[layer], token_loc=tokens
        )
        
        hidden_state = hidden[f"hidden_{layer}"]
        hidden_state0 = hidden_state[0]
        hidden_state1 = hidden_state[1]
        
        if not_pair:
            preds = reporter(hidden_state)[0]
        else:
            preds = reporter.predict(hidden_state0, hidden_state1)

        predictions = np.concatenate([
            predictions,
            np.array([preds.detach().cpu().numpy()])
        ])
    return predictions

def reporter_accuracy_from_strs(
    reporter,
    model,
    tokenizer,
    contrast_pairs,
    layer,
    tokens="last"
):
    predictions = reporter_predictions3(
        reporter, model, tokenizer, contrast_pairs, layer, tokens
    )
    accuracy = (predictions > 0).sum() / predictions.shape[0]
    return accuracy