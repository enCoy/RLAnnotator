from torch.autograd import Variable
import torch

USE_CUDA = True
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)

import numpy as np
def get_pretrained_cnn_output_for_action(state, beat_length, model):
    # this one receives state with shape T,
    scg_signal = state[:beat_length]
    other_part = state[beat_length:]
    # add dimension 1 to first and last axes of scg_signal_patch
    scg_signal = np.expand_dims(np.expand_dims(scg_signal, axis=-1), axis=0)
    # take the other part
    processed_scg = torch.squeeze(model(to_tensor(scg_signal, volatile=True)))
    # processed scg will have shape N x OutChannels x RemainingTime
    # flatten the existing two dimensions
    processed_scg = torch.flatten(processed_scg)
    # now concatenate the processed scg with the other part
    processed_state = np.concatenate((processed_scg.data.cpu().numpy(), other_part), axis=0)
    return processed_state

def get_pretrained_cnn_output(state_batch, beat_length, model):
    scg_signal_patch = state_batch[:, :beat_length]
    other_part = state_batch[:, beat_length:]
    # add dimension 1 to the last axis of scg_signal_patch
    scg_signal_batch = np.expand_dims(scg_signal_patch, axis=-1)
    # take the other part
    processed_scg = model(to_tensor(scg_signal_batch, volatile=True))
    # processed scg will have shape N x OutChannels x RemainingTime
    # flatten the last two dimensions
    processed_scg = torch.flatten(processed_scg, start_dim=1)
    # now concatenate the processed scg with the other part
    processed_state = np.concatenate((processed_scg.data.cpu().numpy(), other_part), axis=-1)
    return processed_state

