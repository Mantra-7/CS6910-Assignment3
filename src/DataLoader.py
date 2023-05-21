import numpy as np
import pandas as pd
from typing import *
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define Lang
class Lang:
	def __init__(self, wordList):
		'''
		Language class to store the vocabulary of the language
		and some useful functions to encode and decode words
		'''
		self.char2index = {'A': 0, 'Z': 1, '_': 2, '^':3}
		self.char2count = {}
		self.index2char = {0: 'A', 1: 'Z', 2: '_', 3: '^'}
		self.n_chars = 4

		for word in wordList:
			self.addWord(word)

	def addWord(self, word):
		'''
		Adds a word to the vocabulary
		'''
		for char in word:
			self.addChar(char)

	def addChar(self, char):
		'''
		Adds a character to the vocabulary
		'''
		if char not in self.char2index:
			self.char2index[char] = self.n_chars
			self.char2count[char] = 1
			self.index2char[self.n_chars] = char
			self.n_chars += 1
		elif char != '_' and char != '^' and char != 'A' and char != 'Z':
			self.char2count[char] += 1

	def encode(self, word):
		'''
		Encodes a word into a list of indices
		'''
		encoded = [0] * len(word)
		for i in range(len(word)):
			if word[i] in self.char2index:
				encoded[i] = self.char2index[word[i]]
			else:
				encoded[i] = self.char2index['^']
		return encoded
			
	def decode(self, word):
		'''
		Decodes a list of indices into a word
		'''
		decoded = ''
		for i in range(len(word)):
			if word[i].argmax().item() in self.index2char:
				decoded += self.index2char[word[i].argmax().item()]
			else:
				decoded += '^'
		return decoded
	
	def decode_words(self, words):
		'''
		Decodes a list of indices into a list of words
		'''
		n_words = words.argmax(dim=2).T
		decoded = ['']*len(n_words)
		for i in range(len(n_words)):
			decoded[i] = ''.join([self.index2char[n_words[i][j].item()] for j in range(len(n_words[i]))])

		return decoded

class DataLoader:
	def __init__(self, lang : str, pad : bool = False, max_length : int = 40):
		'''
		Loads the data from the csv files
		'''
		self.train_data = pd.read_csv(f'aksharantar_sampled/{lang}/{lang}_train.csv')
		self.test_data = pd.read_csv(f'aksharantar_sampled/{lang}/{lang}_test.csv')
		self.valid_data = pd.read_csv(f'aksharantar_sampled/{lang}/{lang}_valid.csv')
		
		self.train_data.columns = ['input_seq', 'target_seq']
		self.test_data.columns = ['input_seq', 'target_seq']
		self.valid_data.columns = ['input_seq', 'target_seq']
			
		self.train_data['input_seq'] = self.train_data['input_seq'].apply(lambda x: x + 'Z')
		self.train_data['target_seq'] = self.train_data['target_seq'].apply(lambda x: x + 'Z')
		self.test_data['input_seq'] = self.test_data['input_seq'].apply(lambda x: x + 'Z')
		self.test_data['target_seq'] = self.test_data['target_seq'].apply(lambda x: x + 'Z')
		self.valid_data['input_seq'] = self.valid_data['input_seq'].apply(lambda x: x + 'Z')
		self.valid_data['target_seq'] = self.valid_data['target_seq'].apply(lambda x: x + 'Z')

		if pad:
			# Pad the sequences to the max length
			self.train_data['input_seq'] = self.train_data['input_seq'].apply(lambda x: x + '_'*(max_length - len(x)))
			self.train_data['target_seq'] = self.train_data['target_seq'].apply(lambda x: x + '_'*(max_length - len(x)))
			self.test_data['input_seq'] = self.test_data['input_seq'].apply(lambda x: x + '_'*(max_length - len(x)))
			self.test_data['target_seq'] = self.test_data['target_seq'].apply(lambda x: x + '_'*(max_length - len(x)))
			self.valid_data['input_seq'] = self.valid_data['input_seq'].apply(lambda x: x + '_'*(max_length - len(x)))
			self.valid_data['target_seq'] = self.valid_data['target_seq'].apply(lambda x: x + '_'*(max_length - len(x)))

		self.inp_lang = Lang(self.train_data['input_seq'])
		self.out_lang = Lang(self.train_data['target_seq'])

	def tensorFromWord(self, lang : Lang, word : str):
		'''
		Converts a word into a tensor of indices
		'''
		indexes = lang.encode(word)
		return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

	def tensorsFromPair(self, pair):
		'''
		Converts a pair of words into a pair of tensors
		'''
		input_tensor = self.tensorFromWord(self.inp_lang, pair[0])
		target_tensor = self.tensorFromWord(self.out_lang, pair[1])
		return (input_tensor.unsqueeze(1), target_tensor)

	def tensorsFromPairs(self, pairs, batch_size):
		'''
		Converts a list of pairs of words into a pair of tensors
		'''
		tensors_inp = []
		tensors_out = []
		for pair in pairs:
			tensors_inp.append(self.tensorFromWord(self.inp_lang, pair[0]))
			tensors_out.append(self.tensorFromWord(self.out_lang, pair[1]))
		return torch.cat(tensors_inp, dim=1).view(-1,1,batch_size), torch.cat(tensors_out, dim=1)
