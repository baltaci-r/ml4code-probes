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
from featurizers.parser_utils import parse
import torch.nn.functional as F
import os, torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
import multiprocessing as mp

# Metrics:

def mse(y_pred, y_test):
    with torch.no_grad():
        mse_loss = nn.MSELoss()
        mse = mse_loss(y_pred, y_test)
    return mse.item(), 'MSE'


def acc(y_pred, y_test):
    with torch.no_grad():
        num = y_pred.shape[0]
        # y_pred = F.one_hot(torch.argmax(y_pred, dim=1), len_seq).detach().numpy()
        if y_test.dim() == 1:
            correct = np.sum(y_pred == y_test.detach().numpy())
        else:
            correct = np.sum((y_pred == y_test.detach().numpy()).all(axis=1))
        accuracy = 100 * correct / num
    return accuracy, 'ACC'


# Tasks:
# ---------
class ListMinTask:
    def __init__(self, input_dim, hidden_size):
        self.model = LSTMModel(input_dim, hidden_size)

    def test(self, X_test, y_test):
        # todo: covert softmax output to argmax output and then calcu
        y_pred = torch.squeeze(self.model.forward(X_test))
        y_pred = y_pred.cpu()
        num, len_seq = y_pred.shape
        y_pred = F.one_hot(torch.argmax(y_pred, dim=1), len_seq).detach().numpy()
        return acc(y_pred, y_test)


class DecodingTask:
    def __init__(self, input_dim):
        self.model = LinearRegression(input_dim, 1)

    def test(self, X_test, y_test):
        y_pred = torch.squeeze(self.model.forward(X_test))
        y_pred = y_pred.cpu()
        return mse(y_pred, y_test)


class AdditionTask:
    def __init__(self, input_dim):
        self.model = ConcatLinearRegression(input_dim, 1)

    def test(self, X_test, y_test):
        y_pred = torch.squeeze(self.model.forward(X_test))
        y_pred = y_pred.cpu()
        return mse(y_pred, y_test)


class LogicTask:
    def __init__(self, type, input_dim, hidden_size, output_size=1):
        self.type = type
        self.model = SeqLSTMModel(type, input_dim, hidden_size, output_size)

    def test(self, X_test, y_test):
        y_pred = torch.squeeze(self.model.forward(X_test))
        y_pred = y_pred.cpu()
        if self.type == 'algebraic':
            return mse(y_pred, y_test)
        else:
            y_pred = torch.sigmoid(y_pred)
            y_pred = (y_pred > 0.5).float().detach().numpy()
            return acc(y_pred, y_test)  # acc(y_pred, y_test) # could be more than one true fixme!


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
        # print('out.size()', out.size())
        # print(self.linear)
        # x.shape = (a, b) and nn.Linear(c, c, bias=False)
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
    def __init__(self, config, task, criterion):
        self.config = config
        self.task = task
        self.criterion = criterion

    def train(self, dataset, batch_size, lr, max_epochs, device):
        writer = SummaryWriter(log_dir=os.path.join('runs', self.config))
        print('Training config:', self.config)
        model = self.task.model
        model = torch.nn.DataParallel(model)
        # device = torch.device("cpu")
        model.to(device)
        # print(model.device)
        model.train()
        dataloader = DataLoader(dataset, batch_size=batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        ep = 0
        running_loss = 0.0
        for epoch in tqdm(range(max_epochs)):
            for i, (x, y) in enumerate(dataloader):
                optimizer.zero_grad()
                x = x.to(device)
                y = y.to(device)
                # print('model-device:', next(model.parameters()).device, 'x-device:', x.get_device(), 'y-device:', y.get_device(), )
                y_pred = model(x)
                y_pred = y_pred.squeeze()
                loss = self.criterion(y_pred, y.squeeze())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                ep += 1
                writer.add_scalar(f"Loss/train", loss.item(), epoch)
                if ep % 200 == 0:  # print every 200 mini-batches
                    print('ep:', ep, '\tloss:', running_loss/200)
                    running_loss = 0.0
        model.eval()

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

    def load_data(self, path, results_dir, data_path, configs):
        print('Loading Data..')
        all_data = []

        with open(data_path) as f :
            for dct in tqdm(f):
                dct = json.loads(dct)
                print(dct.keys())

                config = dct['config']

                config_name = '-'.join([f'{k}-{v}' for k, v in config.items()])
                save_path = os.path.join(path, config_name + '.pt')
                results_path = os.path.join(results_dir, config_name + '.json')

                if config_name in configs and not os.path.exists(results_path):
                    if not os.path.exists(save_path):
                        # continue
                        print(f'Preparing Dataset for: {config_name}')
                        data = pd.DataFrame(dct['data']).apply(tuple, axis=1).tolist()
                        # try:
                        prepared_data = self.prepare_data(data, config['task'])
                        prepared_data = {"config": config, "data": prepared_data}
                        print('Saving prepared data..')
                        torch.save(prepared_data, save_path)
                        # except Exception as e:
                        #     print(e)
                        #     pass

                    else:
                        print(f'Loading Dataset for: {config_name}')
                        prepared_data = torch.load(save_path)
                    all_data.append(prepared_data)
                # elif os.path.exists(results_path):
                #     print(f'Results already exist for {config}')

        return all_data

    # def prepare_data(self, raw_data: List[List[Tuple[str, str, Tuple]]]) -> dict:
    def _prepare_data(self, raw_data):
        lang, code, features = raw_data
        if isinstance(code, list):
            parse_result = parse(lang, ' '.join(code))
        else:
            parse_result = parse(lang, code)

        emb_per_tok = self.model.get_embeddings_per_token_av(parse_result)
        emb_per_tok = [emb_per_tok[l] for l in range(self.num_layers) ]
        tokens = [t.str for t in parse_result.toks]
        return emb_per_tok, lang, parse_result.code, tokens, features

    def prepare_data(self, raw_data: pd.DataFrame, task) -> dict:
        output = {
            "lang": [],  # type: List[str]
            "code": [],  # type: str[NumDocs]
            "tokens": [],  # type: List[List[str]] -- tokenized code
            "embeds": [],  # type: Tensor[NumLayers][NumDocs] where Tensor is of shape (NumModelTokens, EmbeddingDim)
            "features": []  # type: List[bool] or List[bool, bool, ..] or List[int] or  List[float]
        }
        num_skipped = 0
        output['embeds'] = [[] for _ in range(self.num_layers)]
        from multiprocessing.pool import ThreadPool as Pool
        with Pool() as pool:
            for result in tqdm(pool.imap(self._prepare_data, raw_data, chunksize=10), total=len(raw_data)):
            # for i, data in enumerate(tqdm(raw_data)):
            #     print(i, end=' ')
                # if i == 10:
                #     break
                # result = self._prepare_data(data)
                emb_per_tok, lang, code, tokens, features = result
                for l in range(self.num_layers):
                    output['embeds'][l].append(emb_per_tok[l])
                output['lang'].append(lang)
                output['code'].append(code)
                output['tokens'].append(tokens)
                output['features'].append(features)

        if task not in ['list_maximum', 'decoding']:
            for l, out in enumerate(output['embeds']):
                output['embeds'][l] = [o.squeeze() for o in output['embeds'][l]]
                output['embeds'][l] = [torch.unsqueeze(o, 0) if o.dim()==1 else o for o in output['embeds'][l] ]
                # for o in output['embeds'][l]:
                #     print(o.shape)
                output['embeds'][l] = pad_sequence(output['embeds'][l], batch_first=True)
                output['embeds'][l] = torch.split(output['embeds'][l], split_size_or_sections=1)
        print(f"Skipped {num_skipped} entries.")
        return output

    def run(self, all_data: dict, num: int, results_path:str):
        """
        Run the task.
        """
        # results = {}

        hidden_size = 128
        epochs = 100  # TODO: reset to 1K
        lr = 1e-2
        bs = 64

        all_results = {}
        for dct in all_data:
            try:
                config = dct['config']
            except Exception as e:
                print(e)
                continue
            config_name = '-'.join([f'{k}-{v}' for k, v in config.items()]) # + f'_ep_{epochs}'
            print(config_name)
            data = dct['data']
            sub_task_name = config['task']
            type_name = config['type']
            if type_name == 'words':  # dash is causing tokens to be tokenized as seperate, fix later!
                continue

            inputs = data['embeds'][:num]
            num_layers = len(inputs)
            y = data['features'][:num]  # todo: also add to data generation?
            results = {}
            # results[config_name] = {}
            save_path = os.path.join(results_path, config_name + '.json')
            if not os.path.exists(save_path):
                # print('Skipping', config_name)
                # continue

                for l in tqdm(range(num_layers)):
                    config.update({'hs': hidden_size, 'lr': lr, 'bs': bs, 'eps': epochs, 'lay': l})
                    config_name_lay = '-'.join([f'{k}-{v}' for k, v in config.items()])
                    X = torch.vstack(inputs[l])  # fixme: fails at addition, and at logic operation since we need to pad the sequences
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
                            _, output_size = y.shape
                            criterion = nn.BCEWithLogitsLoss()
                            task_model = LogicTask('comparative', input_dim, hidden_size, output_size)
                    else:
                        raise Exception('Undefined Task')

                    # cluster
                    pipe = Pipeline(config_name_lay, task_model, criterion)  # model, criterion, epochs, lr)
                    dataset = TensorDataset(X_train, y_train)
                    device = torch.device("cuda")
                    pipe.train(dataset, bs, lr, epochs, device)
                    X_test = X_test.to(device)
                    print( 'x-device:', X_test.get_device(), 'y-device:', X_test.get_device(), )
                    metric, metric_name = pipe.test(X_test, y_test)

                    print(f"{config_name_lay} layer {l}: {metric_name} : {metric}")

                    # results[config_name][l] = {
                    #     f"{metric_name}": np.round(metric, 2),
                    # }
                    results[l] = {
                        f"{metric_name}": metric,  #np.round(metric, 2),
                    }

                print("=" * 80)
                all_results[config_name] = results
                print(f'Saving to {save_path}')
                with open(save_path, "w") as f:
                    json.dump(results, f, indent=4)

            # else:
            #     with open(save_path, "r") as f:
            #         results = json.load(f)
            #     all_results[config_name] = results

        return all_results
