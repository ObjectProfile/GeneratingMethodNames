import constants
from encoder import EncoderRNN
from decoder import AttnDecoderRNN
from util import time_str
from logger import log, write_training_log, save_dataframe, plot_and_save_histories

import time
import random
from collections import OrderedDict

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size,
                 learning_rate, teacher_forcing_ratio, device):
        super(Seq2Seq, self).__init__()

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = device

        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = AttnDecoderRNN(hidden_size, output_size)

        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)

        self.criterion = nn.NLLLoss()


    def train(self, input_tensor, target_tensor, max_length=constants.MAX_LENGTH):
        encoder_hidden = self.encoder.initHidden()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(max_length + 1, self.encoder.hidden_size, device=self.device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[constants.SOS_TOKEN]], device=self.device)
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di] # Teacher forcing
        else:
            # Without teacher forcing: use its own prediction as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach() # detach from history as input

                loss += self.criterion(decoder_output, target_tensor[di])

                if decoder_input.item() == constants.EOS_TOKEN:
                    break

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / target_length


    def trainIters(self, pairs, first_iter, last_iter, evaluator):
        start_total_time = time.time()
        start_epoch_time = time.time() # Reset every LOG_EVERY iterations
        start_train_time = time.time() # Reset every LOG_EVERY iterations
        total_loss = 0                 # Reset every LOG_EVERY iterations

        if first_iter > 1:
            histories = pd.read_csv(constants.HISTORIES_FILE, sep='\t')
        else:
            histories = pd.DataFrame(
                columns=['Loss', 'BLEU', 'ROUGE', 'F1', 'num_names'])

        avg_loss_history = histories['Loss'].tolist()
        avg_bleu_history = histories['BLEU'].tolist()
        avg_rouge_history = histories['ROUGE'].tolist()
        avg_f1_history = histories['F1'].tolist()
        num_unique_names_history = histories['num_names'].tolist()


        for iter in range(first_iter, last_iter + 1):
            training_pair = random.choice(pairs)
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = self.train(input_tensor, target_tensor)
            total_loss += loss

            if iter % constants.LOG_EVERY == 0:
                train_time_elapsed = time.time() - start_train_time

                torch.save(self.state_dict(), constants.TRAINED_MODEL_FILE)

                with open(constants.ITERS_COMPLETED_FILE, 'w') as f:
                    f.write(str(iter))

                start_eval_time = time.time()
                names = evaluator.evaluate(self)
                eval_time_elapsed = time.time() - start_eval_time

                histories.append({
                    'Loss': total_loss / constants.LOG_EVERY,
                    'BLEU': names['BLEU'].mean(),
                    'ROUGE': names['ROUGE'].mean(),
                    'F1': names['F1'].mean(),
                    'num_names': len(names['Our Name'].unique())
                }, ignore_index=True)

                epoch_time_elapsed = time.time() - start_epoch_time
                total_time_elapsed = time.time() - start_total_time

                histories_last_row = histories.iloc[-1]

                log_dict = OrderedDict([
                    ("Iteration",  '{}/{} ({:.1f}%)'.format(iter, last_iter, iter / last_iter * 100)),
                    ("Average loss", histories_last_row['Loss']),
                    ("Average BLEU", histories_last_row['BLEU']),
                    ("Average ROUGE", histories_last_row['ROUGE']),
                    ("Average F1", histories_last_row['F1']),
                    ("Unique names", histories_last_row['num_names']),
                    ("Epoch time", time_str(epoch_time_elapsed)),
                    ("Training time", time_str(train_time_elapsed)),
                    ("Evaluation time", time_str(eval_time_elapsed)),
                    ("Total training time", time_str(total_time_elapsed))
                ])

                write_training_log(log_dict, constants.TRAIN_LOG_FILE)
                plot_and_save_histories(histories)
                save_dataframe(names, constants.VALIDATION_NAMES_FILE)
                save_dataframe(histories, constants.HISTORIES_FILE)

                # Reseting counters
                total_loss = 0
                start_epoch_time = time.time()
                start_train_time = time.time()


    def forward(self, input_tensor, max_length=constants.MAX_LENGTH, return_attention=False):
        encoder_hidden = self.encoder.initHidden()

        input_length = input_tensor.size(0)

        encoder_outputs = torch.zeros(max_length + 1, self.encoder.hidden_size, device=self.device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[constants.SOS_TOKEN]], device=self.device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        attention_vectors = []

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)

            decoded_words.append(topi.item())
            attention_vectors.append(decoder_attention.tolist()[0])

            if decoded_words[-1] == constants.EOS_TOKEN:
                break

            decoder_input = topi.squeeze().detach()

        if return_attention:
            return decoded_words, attention_vectors
        else:
            return decoded_words
