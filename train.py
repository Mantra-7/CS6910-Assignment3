import wandb
import argparse
from src.Translator import Translator

parser = argparse.ArgumentParser(description='Train a translator')

parser.add_argument('-wp', '--wandb_project', type=str, default='myprojectname', help='Project name used to track experiments in Weights & Biases dashboard')
parser.add_argument('-we', '--wandb_entity', type=str, default='myname', help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
parser.add_argument('-l', '--language', type=str, default='guj', help='choices:  ["asm", "ben", "brx", "guj", "hin", "kan", "kas", "kok", "mai", "mal", "mar", "mni", "ori", "pan", "san", "sid", "tam", "tel", "urd"]')
parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train neural network.')
parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size to train neural network.')
parser.add_argument('-es', '--embed_size', type=int, default=10, help='Embedding size of neural network.')
parser.add_argument('-hs', '--hidden_size', type=int, default=10, help='Hidden size of neural network.')
parser.add_argument('-el', '--enc_layers', type=int, default=1, help='Number of layers in encoder.')
parser.add_argument('-dl', '--dec_layers', type=int, default=1, help='Number of layers in decoder.')
parser.add_argument('-ml', '--max_length', type=int, default=50, help='Maximum length of input sequence.')
parser.add_argument('-t', '--type', type=str, default='gru', help='Type of RNN cell. choices: ["rnn", "gru", "lstm"]')
parser.add_argument('-d', '--dropout', type=float, default=0.2, help='Dropout probability.')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate of neural network.')
parser.add_argument('-o', '--optimizer', type=str, default='sgd', help='Optimizer to use. choices: ["sgd", "adam"]')
parser.add_argument('-a', '--is_attn', action='store_true', help='Whether to use attention mechanism or not.')
parser.add_argument('-log', '--log', action='store_true', help='Whether to log or not.')
parser.add_argument('-dn', '--dumpName', type=str, default='models/model', help='Name of the file to dump the model.')
parser.add_argument('-pfn', '--pred_file_name', type=str, default='predictions', help='Name of the file to dump the predictions.')

args = parser.parse_args()

translator = Translator(args.language, args.embed_size, args.hidden_size, args.enc_layers, args.dec_layers, args.max_length, args.type, args.optimizer, args.dropout, args.batch_size, args.is_attn)

wandb.init(project=args.wandb_project, entity=args.wandb_entity)

translator.train(epoch = args.epochs, learning_rate = args.learning_rate, print_every=10000, log = args.log, wandb = wandb, dumpName = 'models/' + args.dumpName)

test_stat = translator.calculate_stats(translator.dl.test_data)

print("Test Accuracy: ", test_stat[1])
print("Test Loss: ", test_stat[0])

if args.log:
	wandb.log({'Test Accuracy': test_stat[1], 'Test Loss': test_stat[0]})

	pred_file = open(args.pred_file_name + '.csv', 'w')
	for w in translator.dl.test_data['input_seq']:
		out = translator.translate(w)
		pred_file.write(w + ',' + out + '\n')

	pred_file.close()