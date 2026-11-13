# Part 1 · Weight space on Qwen3.5-27B

Two guided notebooks that move Calderwood's memory into the weights of **Qwen3.5-27B** with LoRA
(Unsloth), written for Google Colab on one A100 80GB GPU.

| # | Notebook | What it does | Data it needs |
|---|---|---|---|
| 01 | [From agent memory traces to weights](01_Qwen3_5_27B_Calderwood_memory_traces_guided.ipynb) | Turns the traces into training rows, explains LoRA from a single weight up, trains one adapter and scores the 61 held-out questions before and after. The notebook reports its first run (9 November 2026, one A100 80GB): 13% to 82% after 7 minutes of training, with 0.32% of the parameters trained | The four `.jsonl` files in this folder |
| 02 | [Continual learning in weight space](02_Qwen3_5_27B_Calderwood_continual_learning_guided.ipynb) | Repeats the cycle over three rounds of Calderwood's knowledge, naive against replay, and reads the retention grid, average accuracy and backward transfer | `calderwood_data_v2.zip` (the `rounds/` files for rounds 1 to 3), **not in this repository yet** |

## Running them

**In Colab:** open a notebook, choose an A100 runtime, and upload the data to the Colab files panel
before the setup cell: the four `.jsonl` files from this folder for notebook 01 (or a zip of them
named `calderwood_data.zip`), and `calderwood_data_v2.zip` for notebook 02. The setup cell checks
that the files are there.

**On your own NVIDIA GPU:** run notebook 01 from this folder; its data files are beside it. The
setup cell installs Unsloth with a CUDA build of PyTorch, so these notebooks don't run on a Mac. For a version that runs on an Apple
Silicon laptop with a 0.6B model, see [`../beyond_the_context_window/`](../beyond_the_context_window/).

## The data

`traces.jsonl`, `memory_units.jsonl`, `train.jsonl` and `eval.jsonl` are the same files as
[`../data/calderwood/`](../data/calderwood/), where their generator and a description of each file
are. They are copied here so each run finds them in its working folder.
