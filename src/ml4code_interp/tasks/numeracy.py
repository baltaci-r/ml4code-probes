"""
This module contains the code for probing numerical and logical reasoning capabilities of LMs.
"""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from typing import List, Tuple
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from ml4code_interp.featurizers.parser_utils import parse
import torch.nn.functional as F
import os, torch
import torch.nn as nn


# Metrics:

def mse(y_pred, y_test):
    with torch.no_grad():
        # Calculating the loss and accuracy for the test dataset
        
        mse_loss = nn.MSELoss()
        mse = mse_loss(y_pred, y_test)
    return mse, 'MSE'
    
def acc(y_pred, y_test):
    with torch.no_grad():
        num, len_seq = y_pred.shape
        y_pred = F.one_hot(torch.argmax(y_pred, dim=1), len_seq).detach().numpy()
        correct = np.sum((y_pred == y_test.detach().numpy()).all(axis=1))
        accuracy = 100 * correct / num
    return accuracy, 'ACC'

# Tasks:
# ---------
class ListMinTask:
    def __init__(self, input_dim, hidden_size):
        self.model = LSTMModel(input_dim, hidden_size)

    def test(self, X_test, y_test):
        y_pred = torch.squeeze(self.model.forward(X_test))
        return acc(y_pred, y_test)

class DecodingTask:
    def __init__(self, input_dim):
        self.model = LinearRegression(input_dim, 1)

    def test(self, X_test, y_test):
        y_pred = torch.squeeze(self.model.forward(X_test))
        return mse(y_pred, y_test)


class AdditionTask:
    def __init__(self, input_dim):
        self.model = ConcatLinearRegression(input_dim, 1)

    def test(self, X_test, y_test):
        y_pred = torch.squeeze(self.model.forward(X_test))
        return mse(y_pred, y_test)


class LogicTask:
    def __init__(self, type, input_dim, hidden_size, output_size=1):
        self.type = type
        self.model = SeqLSTMModel(type, input_dim, hidden_size, output_size)

    def test(self, X_test, y_test):
        y_pred = torch.squeeze(self.model.forward(X_test))
        if self.type == 'algebraic':
            return mse(y_pred, y_test)
        else:
            return 0, '' # acc(y_pred, y_test) # could be more than one true fixme!

# Models:
# ---------

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        # self.sigmoid = torch.nn.Sigmoid() # BCEWithLogitsLoss more stable

    def forward(self, x):
        out, hidden = self.lstm(x)  # x: bs, len_seq, embedding_dim, out[0]: bs, len_seq, hidden_size
        out = self.linear(out)  # bs, len_seq, hidden_size -> bs, len_seq, 1 -> sigmoid
        return out


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


class ConcatLinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConcatLinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # concat before LR
        bs = x.shape[0]
        x = x.view(bs, -1)
        out = self.linear(x)
        return out


class SeqLSTMModel(nn.Module):
    def __init__(self, type, input_dim, hidden_size, output_size):
        super(SeqLSTMModel, self).__init__()
        self.type = type
        self.lstm = nn.LSTM(input_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        # self.sigmoid = torch.nn.Sigmoid() # BCEWithLogitsLoss more stable

    def forward(self, x):
        out, hidden = self.lstm(x)
        out = self.linear(out[:, -1])
        return out


# Pipeline:
# ---------
class Pipeline:
    def __init__(self, task, criterion):
        # self.model = model
        self.task = task
        self.criterion = criterion

    def train(self, dataset, batch_size, lr, max_epochs):
        self.task.model.train()
        dataloader = DataLoader(dataset, batch_size=batch_size)
        optimizer = torch.optim.SGD(self.task.model.parameters(), lr=lr)
        for epoch in tqdm(range(max_epochs)):
            running_loss = 0.0
            for i, (x, y) in enumerate(dataloader):
                optimizer.zero_grad()
                y_pred = self.task.model(x).squeeze()
                loss = self.criterion(y_pred, y.squeeze())  # change list maximum to the original 1, 0 binary ex: [0, 1, 0, 0, 0]
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 0:  # print every 2000 mini-batches
                    running_loss = 0.0

    def test(self, X_test, y_test):
        return self.task.test(X_test, y_test)


class NumeracyTask:
    """
    This class implements the numerical and logical reasoning task.
    """

    def __init__(self, model):
        """
        Initialize the task.
        :param model: The model to use.
        """
        self.model = model
        self.num_layers = model.model.config.num_hidden_layers

    def load_data(self, path, data_path):
        # todo: convert data to one json file
        if not os.path.exists(path):
            with open(data_path) as f:
                all_data = []
                for dct in tqdm(f):
                    dct = json.loads(dct)
                    config = dct['config']
                    data = pd.DataFrame(dct['data']).apply(tuple, axis=1).tolist()
                    if config['task'] in ['list_maximum', 'decoding']:
                        prepared_data = self.prepare_data_seperate(data)
                    else:
                        prepared_data = self.prepare_data_series(data)   # todo: change all to list decoding to list of one value, logic to list of bools.. better chaneg all to lists!# todo: remove dependancy on task here and jsut genrewate data that fits the datgenration
                    all_data.append({"config": config, "data": prepared_data})
            torch.save(all_data, path)
        return torch.load(path)

    # def prepare_data(self, raw_data: List[List[Tuple[str, str, Tuple]]]) -> dict:
    def prepare_data_seperate(self, raw_data: pd.DataFrame) -> dict:
        output = {
            "lang": [],  # type: List[str]
            "code": [],  # type: str[NumDocs]
            "tokens": [],  # type: List[List[str]] -- tokenized code
            "embeds": [],  # type: Tensor[NumLayers][NumDocs] where Tensor is of shape (NumModelTokens, EmbeddingDim)
            "features": []  # type: List[bool] or List[bool, bool, ..] or List[int] or  List[float]
        }

        num_skipped = 0
        output['embeds'] = [[] for _ in range(self.num_layers)]
        for i, (lang, code, features) in tqdm(enumerate(raw_data)):  # JUST FOR NOW, later also add the args.num and assert it is less than the genreate dataset size and experiment
            if isinstance(code, list):
                parse_result = parse(lang, ' '.join(code))
            else:
                parse_result = parse(lang, code)
            emb_per_tok = self.model.get_embeddings_per_token_av(parse_result)

            for l in range(self.num_layers):
                output['embeds'][l].append(emb_per_tok[l])
            output['lang'].append(lang)
            output['code'].append(parse_result.code)
            output['tokens'].append([t.str for t in parse_result.toks])  # in the words mode it is splitting further, floats and innts working well!
            output['features'].append(features)
        print(f"Skipped {num_skipped} entries.")
        return output

    def prepare_data_series(self, raw_data):
        output = {
            "lang": [],  # type: List[str]
            "code": [],  # type: str[NumDocs]
            "tokens": [],  # type: List[List[str]] -- tokenized code
            "embeds": [],  # type: Tensor[NumLayers][NumDocs] where Tensor is of shape (NumModelTokens, EmbeddingDim)
            "features": []  # type: List[bool] or List[bool, bool, ..] or List[int] or  List[float]
        }
        output['embeds'] = [[] for _ in range(self.num_layers)]
        for i, (lang, code, features) in tqdm(enumerate(raw_data)):
            if isinstance(code, list):
                parse_result = parse(lang, ' '.join(code))
            else:
                parse_result = parse(lang, code)
            output['lang'].append(lang)
            output['code'].append(parse_result.code)
            output['tokens'].append([t.str for t in parse_result.toks])
            output['features'].append(features)
        output['embeds'] = self.model.get_embeddings_batch(output['code']) # we better use no order information with addition! fix later (i.e use parseresult and take out unuseful start and end toekns)
        return output

    def run(self, all_data: dict, num: int):
        """
        Run the task.
        """
        results = {}

        hidden_size = 128
        epochs = 100
        lr = 1e-4
        bs = 128
        
        for dct in all_data:  # [8:]:  # JUST FOR NOW
            config = dct['config']
            data = dct['data']
            config_name = '-'.join([f'{k}-{v}' for k, v in config.items()])
            sub_task_name = config['task']
            type_name = config['type']
            if type_name == 'words':  # dash is causing tokens to be tokenized as seperate, fix later!
                continue

            inputs = data['embeds'][:num]
            num_layers = len(inputs)
            y = data['features'][:num]  # todo: also add to data generation?
            results[config_name] = {}

            for l in tqdm(range(num_layers)):
                if l==1:
                    break
                X = torch.vstack(inputs[l])  # fails at logic operation since we need to pad the sequences
                y = Tensor(y)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.5, random_state=42
                )

                if sub_task_name == 'list_maximum':
                    criterion = nn.BCEWithLogitsLoss()
                    input_dim = X_train.shape[-1]
                    task_model = ListMinTask(input_dim, hidden_size)
                elif sub_task_name == 'decoding':
                    criterion = nn.MSELoss()
                    input_dim = X_train.shape[-1]
                    task_model = DecodingTask(input_dim)
                elif sub_task_name == 'addition':
                    # alot of nan's, inf's, .. and very large numbers, fixme!
                    criterion = nn.MSELoss()
                    _, len_seq, input_dim = X_train.shape
                    task_model = AdditionTask(input_dim*len_seq)
                elif sub_task_name == 'logic':
                    input_dim = X_train.shape[-1]
                    if 'algebraic' in type_name:
                        criterion = nn.MSELoss()  
                        task_model = LogicTask('algebraic', input_dim, hidden_size)
                    elif 'bool' in type_name:
                        criterion = nn.BCEWithLogitsLoss()  
                        task_model = LogicTask('bool', input_dim, hidden_size)
                    elif 'comparative' in type_name:
                        _, output_size = y.shape  # todo
                        criterion = nn.BCEWithLogitsLoss()   # fixme: could be multiple true.. multi!
                        task_model = LogicTask('comparative', input_dim, hidden_size, output_size)
                else:
                    raise('Undefined Task')
                
                pipe = Pipeline(task_model, criterion)  # model, criterion, epochs, lr)
                dataset = TensorDataset(X_train, y_train)
                pipe.train(dataset, bs, lr, epochs)
                metric, metric_name = pipe.test(X_test, y_test)

                print(f"{config_name} layer {l}: {metric_name} : {metric}")

                results[config_name][l] = {
                    f"{metric_name}": metric,
                }

            print("=" * 80)

        return results
