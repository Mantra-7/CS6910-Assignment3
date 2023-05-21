import math
import time
import random
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.DataLoader import DataLoader
from src.Model import Seq2Seq

def asMinutes(s):
    '''
	Creates a string of minutes and seconds from a number of seconds
	'''
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    '''
	Creates a string of time since a certain time
	'''
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def get_optim(optims, model, lr):
	'''
	Returns the optimizer
	'''
	if optims == 'sgd':
		return optim.SGD(model.parameters(), lr=lr)
	elif optims == 'adam':
		return optim.Adam(model.parameters(), lr=lr)
	elif optims == 'rmsprop':
		return optim.RMSprop(model.parameters(), lr=lr)

class Translator:
	'''
	Translator class to translate the input sequence to the target sequence
	'''
	def __init__(self, lang, embed_size=10, hidden_size=10, enc_layers=1, dec_layers=1, max_length=50, type='gru', optim='sgd', dropout=0.2, batch_size=1, is_attn=False):
		'''
		Initializes the translator class
		'''
		if batch_size != 1:
			self.dl = DataLoader(lang, pad = True, max_length = max_length)
		else:
			self.dl = DataLoader(lang, pad = False, max_length = max_length)

		self.inp_lang = self.dl.inp_lang
		self.out_lang = self.dl.out_lang
		self.batch_size = batch_size

		self.model = Seq2Seq(self.inp_lang.n_chars, hidden_size, embed_size, self.out_lang.n_chars, enc_layers, dec_layers, type, dropout, batch_size, max_length, is_attn)
		self.criterion = nn.NLLLoss()
		self.max_length = max_length
		self.optim = optim
		self.encoder_optim = get_optim(optim, self.model.encoder, 0.001)
		self.decoder_optim = get_optim(optim, self.model.decoder, 0.001)

		ps = [(self.dl.train_data['input_seq'][i], self.dl.train_data['target_seq'][i]) for i in range(len(self.dl.train_data))]
		left = len(ps) % batch_size
		if left != 0:
			ps = ps[:-left]
		self.pairs = [self.dl.tensorsFromPairs(ps[x:x+batch_size], self.batch_size) for x in range(0, len(ps), batch_size)]

	def trainOne(self, input_tensor, target_tensor):
		'''
		Trains the model for one batch
		'''
		self.encoder_optim.zero_grad()
		self.decoder_optim.zero_grad()

		bs = input_tensor.size(2)

		decoder_outputs = self.model.forward(input_tensor, target_tensor)

		loss = 0
		if self.batch_size != 1:
			for i in range(bs):
				loss1 = 0
				for j in range(len(decoder_outputs)):
					loss1 += self.criterion(decoder_outputs[j][i], target_tensor[j][i]) / bs
					if target_tensor[j][i].item() == 1:
						loss += loss1 * self.max_length / (j+1)
						loss1 = 0
						break
				loss += loss1
		else:
			for di in range(len(decoder_outputs)):
				loss += self.criterion(decoder_outputs[di], target_tensor[di]) 
		loss.backward()

		self.encoder_optim.step()
		self.decoder_optim.step()

		return loss.item() / target_tensor.size(0)

	def train(self,epoch=1, n_iters=10000, print_every=1000, plot_every=100, learning_rate=0.01, rand=False, dumpName='model', log=False, wandb = None):
		'''
		Trains the model for some epochs or n_iters iterations
		'''
		self.encoder_optim = get_optim(self.optim, self.model.encoder, learning_rate)
		self.decoder_optim = get_optim(self.optim, self.model.decoder, learning_rate)

		start = time.time()
		train_loss = []
		train_acc = []
		valid_loss = []
		valid_acc = []

		for i in range(epoch):
			print_loss_total = 0
			tot_loss = 0
			print("Epoch: ", i)
			if rand:
				training_pairs = [random.choice(self.pairs) for i in range(n_iters)]
			else:
				training_pairs = self.pairs

			for iter in tqdm(range(1, len(training_pairs) + 1)):
				training_pair = training_pairs[iter - 1]
				input_tensor = training_pair[0]
				target_tensor = training_pair[1]

				loss = self.trainOne(input_tensor, target_tensor)
				print_loss_total += loss 
				tot_loss += loss

				if iter % print_every == 0:
					print_loss_avg = print_loss_total / print_every
					print_loss_total = 0
					print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
												iter, iter / n_iters * 100, print_loss_avg))
			train_loss.append(tot_loss / len(training_pairs))
			train_acc.append(self.accuracy(self.train_data))
			valid_stats = self.calculate_stats(self.valid_data)
			valid_loss.append(valid_stats[0])
			valid_acc.append(valid_stats[1])
			print("Train Loss: ", train_loss[-1], "Valid Loss: ", valid_loss[-1], "Train Acc: ", train_acc[-1], "Valid Acc: ", valid_acc[-1])
			if log:
				wandb.log({'train_loss': train_loss[i], 'train_accuracy': train_acc[i], 'val_loss': valid_loss[i], 'val_accuracy': valid_acc[i]})
		pickle.dump(self, open(dumpName + '.pkl', 'wb'))
		return train_loss, train_acc, valid_loss, valid_acc	

	def accuracy(self, data):
		'''
		Calculates the accuracy of the model on the given data
		'''
		with torch.no_grad():
			acc = 0
			if self.batch_size == 1:
				for i in range(0, len(data), self.batch_size):
					tensors = self.dl.tensorsFromPairs([(data['input_seq'][j], data['target_seq'][j]) for j in range(i, i + self.batch_size)], self.inp_lang, self.out_lang, self.batch_size)
					outputs = self.model.forward(tensors[0], tensors[1])
					words = self.out_lang.decode_words(outputs)
					acc += np.sum([words[j].strip('Z') == data['target_seq'][i + j].strip('Z_') for j in range(self.batch_size)])
			else:
				tensors = self.dl.tensorsFromPairs([(data['input_seq'][j], data['target_seq'][j]) for j in range(len(data))], self.inp_lang, self.out_lang, len(data))
				outputs = self.model.forward(tensors[0], tensors[1])
				words = self.out_lang.decode_words(outputs)
				acc += np.sum([words[j].strip('Z') == data['target_seq'][j].strip('Z_') for j in range(len(data))])
			return acc / len(data)

	def calculate_stats(self, data):
		'''
		Calculates the loss and accuracy of the model on the given data
		'''
		with torch.no_grad():
			loss = 0
			acc = 0
			if self.batch_size == 1:
				for i in range(0, len(data), self.batch_size):
					tensors = self.dl.tensorsFromPairs([(data['input_seq'][j], data['target_seq'][j]) for j in range(i, i + self.batch_size)], self.inp_lang, self.out_lang, self.batch_size)
					outputs = self.model.forward(tensors[0], tensors[1])
					for di in range(len(outputs)):
						loss += self.criterion(outputs[di], tensors[1][di]) / tensors[1].size(0)
					words = self.out_lang.decode_words(outputs)
					acc += np.sum([words[j].strip('Z') == data['target_seq'][i + j].strip('Z_') for j in range(self.batch_size)])
			else:
				tensors = self.dl.tensorsFromPairs([(data['input_seq'][j], data['target_seq'][j]) for j in range(len(data))], self.inp_lang, self.out_lang, len(data))
				outputs = self.model.forward(tensors[0], tensors[1])
				for i in range(len(data)):
					loss1 = 0
					for j in range(len(outputs)):
						loss1 += self.criterion(outputs[j][i], tensors[1][j][i])
						if tensors[1][j][i].item() == 1:
							loss += loss1 / (j+1)
							loss1 = 0
							break
					loss += loss1 / len(outputs)
				words = self.out_lang.decode_words(outputs)
				acc += np.sum([words[j].strip('Z') == data['target_seq'][j].strip('Z_') for j in range(len(data))])
			return loss.item() / len(data), acc / len(data)

	def translate(self, word):
		'''
		Translates the given word
		'''
		with torch.no_grad():
			if self.batch_size != 1:
				word = word + 'Z' + '_'*(self.max_length - len(word) -1)
			tensor = self.dl.tensorFromWord(self.inp_lang, word).unsqueeze(1)
			outs = self.model.predict(tensor)
			return self.out_lang.decode(outs)