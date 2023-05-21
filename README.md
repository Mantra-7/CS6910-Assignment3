# CS6910 - Fundamentals of Deep Learning - Assignment 3
### Author : Mantra Trambadia (CS20B083)

This repository contains the code for the third assignment of the course CS6910 - Fundamentals of Deep Learning. In this assignment, I have implemented a translator from English to Indian languages using a sequence-to-sequence model with attention. The model is trained on the [Aksharantar dataset](https://drive.google.com/file/d/1uRKU4as2NlS9i8sdLRS1e326vQRdhvfw/view?usp=share_link).

## Code structure

`src` folder contains the following files:
- `DataLoader.py` : Contains the class `DataLoader` which is used to load the data.
- `Model.py` : Contains the encoder, decoder and `Seq2Seq` which is model for training.
- `Translator.py` : Contains the class `Translator` which is provides high level interface for training and testing the model.

`train.py` is the main file which is used to train the neural network.

## Running the code

```bash
python train.py [ARGUMENTS]
```

### Supported arguments

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | myname | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-l`, `--language` | guj | choices:  ["asm", "ben", "brx", "guj", "hin", "kan", "kas", "kok", "mai", "mal", "mar", "mni", "ori", "pan", "san", "sid", "tam", "tel", "urd"] |
| `-e`, `--epochs` | 10 | Number of epochs to train neural network. |
| `-b`, `--batch_size` | 1 | Batch size to train neural network. |
| `-es`, `--embed_size` | 10 | Embedding size of neural network. |
| `-hs`, `--hidden_size` | 10 | Hidden size of neural network. |
| `-el`, `--enc_layers` | 1 | Number of layers in encoder. |
| `-dl`, `--dec_layers` | 1 | Number of layers in decoder. |
| `-ml`, `--max_length` | 50 | Maximum length of input sequence. |
| `-t`, `--type` | gru | Type of RNN cell. choices: ["rnn", "gru", "lstm"] |
| `-d`, `--dropout` | 0.2 | Dropout probability. |
| `-lr`, `--learning_rate` | 0.001 | Learning rate of neural network. |
| `-o`, `--optimizer` | sgd | Optimizer to use. choices: ["sgd", "adam"] |
| `-a`, `--is_attn` | False | Whether to use attention mechanism or not. |
| `-log`, `--log` | False | Whether to log or not. |
| `-dn`, `--dumpName` | models/model | Name of the file to dump the model. |
| `-pfn`, `--pred_file_name` | predictions | Name of the file to dump the predictions. |

## Using my classes

You can use my classes to train your translator. The following code snippet shows how to use my classes to train a Seq2Seq translator.

```python
from src.Translator import Translator

translator = Translator(language, embed_size, hidden_size, enc_layers, dec_layers, max_length, type, optimizer, dropout, batch_size, is_attn)

train_loss, train_acc, val_loss, val_acc = translator.train(epoch = epochs, learning_rate = learning_rate, print_every=10000, log = log, wandb = wandb, dumpName = 'models/' + dumpName)

test_loss, test_acc = translator.calculate_stats(translator.dl.test_data)
```

You can make your own custom model in `Model.py` file similar to `Seq2Seq` class and use it to train your translator.

See `train.py` for more details.