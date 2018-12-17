import constants
from seq2seq import Seq2Seq
from preprocessing import read_langs, train_validation_test_split, TensorBuilder
from evaluator import Evaluator
from logger import log, save_pickle, save_dataframe, use_default_logger, is_drive_logger

import time
import math
from collections import OrderedDict

import numpy as np
import pandas as pd

import torch


def start_training():
    try:
        total_time_start = time.time()

        log('Loading the data')
        methods = pd.read_csv(constants.DATASET_FILE, delimiter='\t')

        if constants.DROP_DUPLICATES:
            log('Removing duplicate methods')
            methods = methods.drop_duplicates()

        log('Building input and output languages')
        input_lang, output_lang, pairs = read_langs('source', 'name', methods)

        log('Number of unique input tokens: {}\n'
            'Number of unique output tokens: {}'.format(
                input_lang.n_words,
                output_lang.n_words
        ))

        log('Serializing input and output languages to pickles')

        save_pickle(input_lang, constants.INPUT_LANG_FILE)
        save_pickle(output_lang, constants.OUTPUT_LANG_FILE)

        log('Splitting data into train, validation, and test sets')

        train_pairs, valid_pairs, test_pairs = train_validation_test_split(
            pairs, constants.TRAIN_PROP, constants.VALID_PROP, constants.TEST_PROP)

        log('Train size: {}\n'
            'Validation size: {}\n'
            'Test size: {}'.format(
                len(train_pairs),
                len(valid_pairs),
                len(test_pairs)
        ))

        log('Serializing train, validation, and test sets to pickles')

        save_pickle(train_pairs, constants.TRAIN_PAIRS_FILE)
        save_pickle(valid_pairs, constants.VALID_PAIRS_FILE)
        save_pickle(test_pairs, constants.TEST_PAIRS_FILE)

        log('Converting data entries to tensors')

        tensor_builder = TensorBuilder(input_lang, output_lang)
        train_pairs = [tensor_builder.tensorsFromPair(pair) for pair in train_pairs]
        valid_pairs = [tensor_builder.tensorsFromPair(pair) for pair in valid_pairs]
        test_pairs = [tensor_builder.tensorsFromPair(pair) for pair in test_pairs]

        log('Building the model')

        model = Seq2Seq(
            input_size=input_lang.n_words,
            output_size=output_lang.n_words,
            hidden_size=constants.HIDDEN_SIZE,
            learning_rate=constants.LEARNING_RATE,
            teacher_forcing_ratio=constants.TEACHER_FORCING_RATIO,
            device=constants.DEVICE)

        log(str(model))

        log('Initializing evaluators')

        evaluator = Evaluator(valid_pairs, input_lang, output_lang)
        test_set_evaluator = Evaluator(test_pairs, input_lang, output_lang)

    except Exception as e:
        # Log the error message and raise it again so see more info
        log("Error: " + str(e))
        raise e

    successful = False
    restarting_attempts = 10
    iters_completed = 0

    while not successful:
        try:
            log('Training the model')
            model.trainIters(train_pairs, iters_completed, constants.NUM_ITER, evaluator)

            log('Saving the model')
            torch.save(model.state_dict(), constants.TRAINED_MODEL_FILE)

            successful = True

            log('Evaluating on test set')
            names = test_set_evaluator.evaluate(model)
            save_dataframe(names, constants.TEST_NAMES_FILE)

            log('Done')

        except Exception as e:
            restarted = False

            while not restarted and restarting_attempts > 0 or is_drive_logger():
                # Log the error message and restart from the last successful state
                try:
                    print('Trying to restart', restarting_attempts)
                    if restarting_attempts == 0:
                        use_default_logger()
                        print("Switched to default logger")
                        restarting_attempts = 5

                    restarting_attempts -= 1

                    log("Error during training: " + str(e))

                    try:
                        with open(constants.ITERS_COMPLETED_FILE, 'r') as f:
                            iters_completed = int(f.read())
                    except Exception as e:
                        log("Error: " + str(e))
                        log("Can't read the number of completed iterations. Starting from 0")

                    log("Restarting from iteration {}\n{} more iterations to go ({:.1f}%)".format(
                        iters_completed + 1,
                        constants.NUM_ITER - iters_completed,
                        (constants.NUM_ITER - iters_completed) / constants.NUM_ITER * 100))

                    try:
                        log('Loading the last trained model')
                        model.load_state_dict(torch.load(constants.TRAINED_MODEL_FILE))
                    except Exception as e:
                        log("Error: " + str(e))
                        log("Can't load the last trained model. Starting from scratch")

                        model = Seq2Seq(
                            input_size=input_lang.n_words,
                            output_size=output_lang.n_words,
                            hidden_size=constants.HIDDEN_SIZE,
                            learning_rate=constants.LEARNING_RATE,
                            teacher_forcing_ratio=constants.TEACHER_FORCING_RATIO,
                            device=constants.DEVICE)

                    restarted = True

                except Exception as e:
                    use_default_logger()
                    print("Switched to default logger")
