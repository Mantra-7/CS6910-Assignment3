import torch
import torch.nn as nn
import torch.nn.functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cell(str):
	'''
	Returns the cell from the type
	'''
	if str == 'lstm':
		return nn.LSTM
	elif str == 'gru':
		return nn.GRU
	elif str == 'rnn':
		return nn.RNN
	else:
		raise ValueError('Invalid cell type')

class EncoderRNN(nn.Module):
	'''
	Encoder class
	'''
	def __init__(self, input_size, embed_size, hidden_size, n_layers=1, type='gru', dropout=0.2, batch_size=1):
		'''
		Initializes the encoder
		'''
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.type_t = type
		self.batch_size = batch_size

		self.embedding = nn.Embedding(input_size, embed_size)
		self.cell = get_cell(type)(embed_size, hidden_size, n_layers, dropout=dropout)

	def forward(self, input, hidden):
		'''
		Forward pass
		'''
		embedded = self.embedding(input)
		output = embedded
		output, hidden = self.cell(output, hidden)
		return output, hidden

	def initHidden(self, bs = -1):
		'''
		Initializes the hidden state
		'''
		if bs == -1:
			bs = self.batch_size
		if self.type_t == 'lstm':
			return torch.zeros(self.n_layers, bs, self.hidden_size, device=device), torch.zeros(self.n_layers, bs, self.hidden_size, device=device)
		else:
			return torch.zeros(self.n_layers, bs, self.hidden_size, device=device)
	
class DecoderRNN(nn.Module):
	'''
	Decoder class
	'''
	def __init__(self, hidden_size, output_size, n_layers=1, type='gru', dropout=0.2, batch_size=1):
		'''
		Initializes the decoder
		'''
		super(DecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.type_t = type
		self.batch_size = batch_size

		self.embedding = nn.Embedding(output_size, hidden_size)
		self.cell = get_cell(type)(hidden_size, hidden_size, n_layers, dropout=dropout)
		self.out = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden, bs = -1):
		'''
		Forward pass
		'''
		if bs == -1:
			bs = self.batch_size
		output = self.embedding(input).view(1, bs, -1)
		output = F.relu(output)
		output, hidden = self.cell(output, hidden)
		output = self.softmax(self.out(output[0]))
		return output, hidden

	def initHidden(self, bs = -1):
		'''
		Initializes the hidden state
		'''
		if bs == -1:
			bs = self.batch_size
		if self.type_t == 'lstm':
			return torch.zeros(self.n_layers, bs, self.hidden_size, device=device), torch.zeros(self.n_layers, bs, self.hidden_size, device=device)
		else:
			return torch.zeros(self.n_layers, bs, self.hidden_size, device=device)
		

class AttnDecoderRNN(nn.Module):
	'''
	Attention decoder class
	'''
	def __init__(self, hidden_size, output_size, n_layers=1, type='gru', dropout_p=0.1, batch_size=1, max_length=50):
		'''
		Initializes the attention decoder
		'''
		super(AttnDecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout_p = dropout_p
		self.max_length = max_length
		self.n_layers = n_layers
		self.type_t = type
		self.batch_size = batch_size

		self.embedding = nn.Embedding(self.output_size, self.hidden_size)
		self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
		self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
		self.dropout = nn.Dropout(self.dropout_p)
		self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, dropout=self.dropout_p)
		self.out = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, input, hidden, encoder_outputs, bs=-1):
		'''
		Forward pass
		'''
		if bs == -1:
			bs = self.batch_size
		embedded = self.embedding(input).view(1, bs, -1)
		embedded = self.dropout(embedded)

		attn_weights = F.softmax(
			self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
		attn_applied = torch.bmm(attn_weights.unsqueeze(0),
								encoder_outputs.unsqueeze(0))

		output = torch.cat((embedded[0], attn_applied[0]), 1)
		output = self.attn_combine(output).unsqueeze(0)

		output = F.relu(output)
		output, hidden = self.gru(output, hidden)

		output = F.log_softmax(self.out(output[0]), dim=1)
		return output, hidden, attn_weights

	def initHidden(self, bs=-1):
		'''
		Initializes the hidden state
		'''
		if bs == -1:
			bs = self.batch_size
		if self.type_t == 'lstm':
			return torch.zeros(self.n_layers, bs, self.hidden_size, device=device), torch.zeros(self.n_layers, bs, self.hidden_size, device=device)
		else:
			return torch.zeros(self.n_layers, bs, self.hidden_size, device=device)

class Seq2Seq(nn.Module):
	'''
	Seq2Seq model class
	'''
	def __init__(self, input_size, hidden_size, embed_size, output_size, enc_layers=1, dec_layers=1, type='gru', dropout=0.2, batch_size=1, max_length=50,is_attn=False):
		'''
		Initializes the seq2seq model
		'''
		super(Seq2Seq, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.enc_layers = enc_layers
		self.dec_layers = dec_layers
		self.batch_size = batch_size
		self.dropout = dropout
		self.max_length = max_length
		self.is_attn = is_attn
		self.type_t = type

		self.encoder = EncoderRNN(input_size, embed_size, hidden_size, enc_layers, type, dropout, batch_size).to(device)
		if not is_attn:
			self.decoder = DecoderRNN(hidden_size, output_size, dec_layers, type, dropout, batch_size).to(device)
		else:
			self.decoder = AttnDecoderRNN(hidden_size, output_size, dec_layers, type, dropout, batch_size, max_length).to(device)

	def forward(self, input_tensor, target_tensor):
		'''
		Forward pass
		'''
		bs = input_tensor.size(2)
		encoder_hidden = self.encoder.initHidden(bs)

		input_length = input_tensor.size(0)
		target_length = target_tensor.size(0)

		encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=device)

		for ei in range(input_length):
			encoder_output, encoder_hidden = self.encoder(
				input_tensor[ei], encoder_hidden)
			encoder_outputs[ei] = encoder_output[0, 0]

		lst = []
		for i in range(bs):
			lst.append([0])
		decoder_input = torch.tensor(lst, device=device)  # SOS

		if self.encoder.n_layers == self.decoder.n_layers:
			decoder_hidden = encoder_hidden
		elif self.type_t != 'lstm':
			decoder_hidden = torch.zeros(self.decoder.n_layers, bs, self.decoder.hidden_size, device=device)
			av = encoder_hidden[0]
			for i in range(1,self.encoder.n_layers):
				av += encoder_hidden[i]
			av /= self.encoder.n_layers
			for i in range(self.decoder.n_layers):
				decoder_hidden[i] = av
		else:
			decoder_hidden = (torch.zeros(self.decoder.n_layers, bs, self.decoder.hidden_size, device=device), torch.zeros(self.decoder.n_layers, bs, self.decoder.hidden_size, device=device))
			av = encoder_hidden[0][0]
			for i in range(1,self.encoder.n_layers):
				av += encoder_hidden[0][i]
			av /= self.encoder.n_layers
			for i in range(self.decoder.n_layers):
				decoder_hidden[0][i] = av
			av = encoder_hidden[1][0]
			for i in range(1,self.encoder.n_layers):
				av += encoder_hidden[1][i]
			av /= self.encoder.n_layers
			for i in range(self.decoder.n_layers):
				decoder_hidden[1][i] = av

		use_teacher_forcing = True if random.random() < 0.5 else False

		decoder_outputs = torch.zeros(target_length, bs, self.output_size, device=device)
		if not self.is_attn:
			if use_teacher_forcing:
				# Teacher forcing: Feed the target as the next input
				for di in range(target_length):
					decoder_output, decoder_hidden = self.decoder(
						decoder_input, decoder_hidden, bs)
					decoder_outputs[di] = decoder_output
					decoder_input = target_tensor[di]  # Teacher forcing
			else:
				# Without teacher forcing: use its own predictions as the next input
				for di in range(target_length):
					decoder_output, decoder_hidden = self.decoder(
						decoder_input, decoder_hidden, bs)
					decoder_outputs[di] = decoder_output
					topv, topi = decoder_output.topk(1)
					decoder_input = topi.squeeze().detach()  # detach from history as input
					if self.batch_size == 1:
						if topi == 1:
							break
		else:
			if use_teacher_forcing:
				# Teacher forcing: Feed the target as the next input
				for di in range(target_length):
					decoder_output, decoder_hidden, decoder_attention = self.decoder(
						decoder_input, decoder_hidden, encoder_outputs, bs)
					decoder_outputs[di] = decoder_output
					decoder_input = target_tensor[di]
			else:
				# Without teacher forcing: use its own predictions as the next input
				for di in range(target_length):
					decoder_output, decoder_hidden, decoder_attention = self.decoder(
						decoder_input, decoder_hidden, encoder_outputs, bs)
					decoder_outputs[di] = decoder_output
					topv, topi = decoder_output.topk(1)
					decoder_input = topi.squeeze().detach()
					if self.batch_size == 1:
						if topi == 1:
							break

		return decoder_outputs

	def predict(self, input_tensor):
		'''
		Predicts the output for the given input
		'''
		encoder_hidden = self.encoder.initHidden(1)

		input_length = input_tensor.size(0)

		encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=device)

		for ei in range(input_length):
			encoder_output, encoder_hidden = self.encoder(
				input_tensor[ei], encoder_hidden)
			encoder_outputs[ei] = encoder_output[0, 0]

		decoder_input = torch.tensor([[0]], device=device)  # SOS

		decoder_outputs = []

		decoder_hidden = encoder_hidden
		if not self.is_attn:
			for di in range(self.max_length):
				decoder_output, decoder_hidden = self.decoder(
					decoder_input, decoder_hidden, 1)
				topv, topi = decoder_output.data.topk(1)
				if topi == 1:
					break
				decoder_outputs.append(decoder_output)
				decoder_input = topi.squeeze().detach()
		else:
			for di in range(self.max_length):
				decoder_output, decoder_hidden, decoder_attention = self.decoder(
					decoder_input, decoder_hidden, encoder_outputs, 1)
				topv, topi = decoder_output.data.topk(1)
				if topi == 1:
					break
				decoder_outputs.append(decoder_output)
				decoder_input = topi.squeeze().detach()

		return decoder_outputs