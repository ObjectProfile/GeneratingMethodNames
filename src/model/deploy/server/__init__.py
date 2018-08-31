from train import Seq2Seq
from train import constants

import os
import pickle

import torch

__all__ = ['load_trained_model']


def __load_langs():
    with open(os.path.join(constants.DATA_DIR, 'input_lang.pkl'), 'rb') as f:
        input_lang = pickle.load(f)

    with open(os.path.join(constants.DATA_DIR, 'output_lang.pkl'), 'rb') as f:
        output_lang = pickle.load(f)

    return input_lang, output_lang


def __build_model(input_lang, output_lang):
    return Seq2Seq(
        input_size=input_lang.n_words,
        output_size=output_lang.n_words,
        hidden_size=constants.HIDDEN_SIZE,
        learning_rate=constants.LEARNING_RATE,
        teacher_forcing_ratio=constants.TEACHER_FORCING_RATIO,
        device=constants.DEVICE)


def __load_model_state(model):
    model.load_state_dict(torch.load(os.path.join(constants.MODELS_DIR, 'trained_model.pt')))
    return model


def load_trained_model():
    input_lang, output_lang = __load_langs()
    model = __build_model(input_lang, output_lang)
    model = __load_model_state(model)
    return model