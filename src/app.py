# import
import gradio as gr
import os
import sys

sys.path.append("../pop2piano")
import glob
import random

import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd
import IPython.display as ipd
import soundfile as sf

from tqdm.auto import tqdm
from omegaconf import OmegaConf
import note_seq

from utils.dsp import get_stereo
from utils.demo import download_youtube
from transformer_wrapper import TransformerWrapper
from midi_tokenizer import MidiTokenizer, extrapolate_beat_times
from preprocess.beat_quantizer import extract_rhythm, interpolate_beat_times

# set config
device = "cuda" if torch.cuda.is_available() else "cpu"
config = OmegaConf.load("../pop2piano/config.yaml")
wrapper = TransformerWrapper(config)
wrapper = wrapper.load_from_checkpoint(
    "../models/model-1999-val_0.67311615.ckpt", config=config
).to(device)
model = "dpipqxiy"
wrapper.eval()

# list parameters
composer_list = [
    "composer1",
    "composer2",
    "composer3",
    "composer4",
    "composer5",
    "composer6",
    "composer7",
    "composer8",
    "composer9",
    "composer10",
    "composer11",
    "composer12",
    "composer13",
    "composer14",
    "composer15",
    "composer16",
    "composer17",
    "composer18",
    "composer19",
    "composer20",
    "composer21",
]


# define function


def dummy_function(composer: str, audio: str) -> str:
    return "Composer: " + composer + " Audio: " + audio


def generate_midi(composer: str, audio: str) -> tuple[str, str,str]:
    pm, composer, mix_path, midi_path = wrapper.generate(
        audio_path=audio,
        composer=composer,
        model=model,
        show_plot=False,
        save_midi=True,
        save_mix=True,
    )
    return mix_path, mix_path, midi_path


# create interface
demo = gr.Interface(
    fn=generate_midi,
    inputs=[
        gr.Dropdown(choices=composer_list, value="composer1", label="Select Composer",),
        gr.Audio(type="filepath", label="Upload Audio", source="upload"),
    ],
    outputs=[gr.Audio(label="Audio Preview"),
             gr.File(label="Download Audio"),
             gr.File(label="Download Midi")],
    title="Pop2Piano Web UI",
    description="This is a web interface for Pop2Piano.",
    theme="gradio/soft"
)

# launch interface
demo.queue()
demo.launch(server_port=8080)
