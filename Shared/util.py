import sys
import argparse
import torch
import torch.nn
import importlib


"""
This file contains utility methods useful across reinforcement learning tasks.
It can create parsers for command line parameters, experience buffers,
and policies. It also has Pytorch utility functions, experience collection
algorithms and so on.
"""


FRAME_RATE = 60
COMMENT_HEADER = "//"

sys.path.append(".")


# Makes a argparse parser from a class of parameter names / default values
def make_parser(params, parser_name):
    fields = vars(params)
    parser = argparse.ArgumentParser(description=parser_name)

    for (attribute, value) in fields.items():
        attribute_fixed = '--' + attribute.replace('_', '-')

        required_type = type(value)
        if required_type == bool:
            required_type = int

        parser.add_argument(
            attribute_fixed, dest=attribute, type=required_type)

    return parser


# Fills a class of parameters with argparse results
def write_parsed_args(params, parsed_args):
    fields = vars(params)
    parsed_fields = vars(parsed_args)

    for (attribute, value) in fields.items():
        if parsed_fields[attribute] is not None:
            to_write = parsed_fields[attribute]
            if type(value) == bool:
                to_write = bool(to_write)
            setattr(params, attribute, to_write)


# Prints out a list of (metric name, value)
def print_metrics(metrics):
    items = []
    for (name, value) in metrics:
        items.append("%s: %s" % (name, value))
    print(", ".join(items))


# Parses metrics printed with this utility module
def parse_metrics(file_name):
    metrics = []

    with open(file_name, 'r') as f:
        for line in f:
            if line.startswith(COMMENT_HEADER):
                continue
            if line.strip() == '':
                continue

            snapshot = {}
            try:
                for metric in line.split(','):
                    (name, value) = metric.split(':')
                    snapshot[name.strip()] = value.strip()
            except ValueError as x:
                print("Error during parsing \"%s\": %s" % (line.strip(), x))
                continue

            metrics.append(snapshot)

    return metrics


# Randomizers based on Pytorch

def torch_rand_fn():
    return torch.rand(1).item()


def torch_randint_fn(high):
    return torch.randint(high, (1,)).item()


# Loads a Pytorch model from a file name inside the models folder
def load_model(model_name):
    module = importlib.import_module("models." + model_name)
    return module.model


# Performs Xavier initialization on a model
def initialize(model):
    for p in model.parameters():
        if p.dim() != 1:
            torch.nn.init.xavier_normal_(p)
        else:
            p.data.zero_()


# Creates a tensor of random integers for Pytorch
def torch_rand_ints(example_tensor, high, size):
    return example_tensor.new_zeros(size, dtype=torch.long).random_(high)


# Turns a function that takes in batches into one that takes in single values
def debatch_function(batch_fn):
    return lambda value: batch_fn(value.unsqueeze(0))[0]
