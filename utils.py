from datetime import datetime
import json
import logging
import numpy as np
import os
import pytz
import torch
import sys
from tqdm import tqdm

logger = logging.getLogger('DSAC.utils')


class Params:
    """
    Class that loads hyperparameters from a json file as a dictionary (also support nested dicts).
    Example:
    params = Params(json_path)
    # access key-value pairs
    params.learning_rate
    params['learning_rate']
    # change the value of learning_rate in params
    params.learning_rate = 0.5
    params['learning_rate'] = 0.5
    # print params
    print(params)
    # combine two json files
    params.update(Params(json_path2))
    """

    def __init__(self, json_path=None):
        if json_path is not None and os.path.isfile(json_path):
            with open(json_path) as f:
                params = json.load(f)
                self.__dict__.update(params)
        else:
            self.__dict__ = {}

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def update(self, json_path=None, params=None):
        """Loads parameters from json file"""
        if json_path is not None:
            with open(json_path) as f:
                params = json.load(f)
                self.__dict__.update(params)
        elif params is not None:
            self.__dict__.update(vars(params))
        else:
            raise Exception('One of json_path and params must be provided in Params.update()!')

    def __contains__(self, item):
        return item in self.__dict__

    def __getitem__(self, key):
        return getattr(self, str(key))

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __str__(self):
        return json.dumps(self.__dict__, sort_keys=True, indent=4, ensure_ascii=False)


def set_logger(log_path):
    """
    Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    logging.info('Starting training...')
    Args:
        log_path: (string) where to log
    """
    _logger = logging.getLogger('DSAC')
    _logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%m/%d %H:%M:%S')

    class TqdmHandler(logging.StreamHandler):
        def __init__(self, formatter):
            logging.StreamHandler.__init__(self)
            self.setFormatter(formatter)
            self.setStream(tqdm)

        def emit(self, record):
            msg = self.format(record)
            tqdm.write(msg)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    _logger.addHandler(file_handler)
    _logger.addHandler(TqdmHandler(fmt))
    # handler = logging.StreamHandler(stream=sys.stdout)
    # _logger.addHandler(handler)

    # https://stackoverflow.com/questions/6234405/logging-uncaught-exceptions-in-python?noredirect=1&lq=1
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            _logger.info('=*=*=*= Keyboard interrupt =*=*=*=')
            return

        _logger.error("Exception --->", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception


def save_dict_to_json(d, json_path):
    """
    Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, global_step, checkpoint, is_best=False):
    """
    Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        global_step: (int) number of updates performed
        checkpoint: (string) folder where parameters are to be saved
        is_best: (boolean)
    """
    if is_best:
        filepath = os.path.join(checkpoint, f'best.pth.tar')
    else:
        filepath = os.path.join(checkpoint, f'latest.pth.tar')
    if not os.path.exists(checkpoint):
        logger.info(f'Checkpoint Directory does not exist! Making directory {checkpoint}')
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    logger.info(f'Checkpoint saved to {filepath}')


def load_checkpoint(file_dir, restore_file, model, optimizer=None, loss=None):
    """
    Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    checkpoint = os.path.join(f'runs/{file_dir}', restore_file + '.pth.tar')
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
    else:
        logger.info(f'Restoring parameters from {checkpoint}')
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location='cuda')
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
    if loss:
        loss = np.load(os.path.join(file_dir, restore_file + '_loss.npy'))
        return checkpoint['global_step'], loss
    else:
        return checkpoint['global_step']


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = np.stack([obs[a] for a in obs], axis=0)
    # transpose to be (batch, channel, height, width)
    obs = obs.transpose(0, -1, 1, 2)
    # convert to torch
    obs = torch.tensor(obs).to(device)

    return obs


def batchify(x, device, key=None):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    if key:
        x = np.stack([x[a][key] for a in x], axis=0)
    else:
        x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x


def name_with_datetime():
    now = datetime.now(tz=pytz.utc)
    now = now.astimezone(pytz.timezone('US/Pacific'))
    return now.strftime("%Y-%m-%d_%H:%M:%S")
