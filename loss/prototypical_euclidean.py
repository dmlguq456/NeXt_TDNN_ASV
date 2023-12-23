from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_metric import accuracy

# Ref: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/proto.py

class LossFunction(nn.Module):
    def __init__(self, nb_sample_per_speaker, nb_speaker, n_shot, **kwargs):
        super(LossFunction, self).__init__()

        self.nb_sample_per_speaker = nb_sample_per_speaker
        self.nb_speaker = nb_speaker
        self.n_shot = n_shot

        self.criterion  = torch.nn.CrossEntropyLoss()



        print('Initialised Prototypical Loss')

    def forward(self, x, label):
        """
        Args:
            x: embedding of each sample
                - shape: (batch, embedding)
            label: label of each sample
                - shape: (batch, )
        Returns:
            nloss: loss
            prec1: accuracy
        """
        BATCH, EMBEDDING = x.shape

        # for debug - check batch configuration
        label = label.reshape(self.nb_speaker, self.nb_sample_per_speaker) # (batch) -> (nb_speaker, nb_sample_per_speaker)
        for i in range(self.nb_sample_per_speaker - 1):
            assert all(label[:,i] == label[:,i+1]) == True, "label should be same for each speaker"

        # to calculate classification accuracy - this is synchronous to query set
        label = label[:, self.n_shot:] # (nb_speaker, nb_sample_per_speaker - n_shot)
        label = label.reshape(self.nb_speaker * (self.nb_sample_per_speaker - self.n_shot)) # (nb_speaker * (nb_sample_per_speaker - n_shot))

        # calculate prototype
        x = x.reshape(self.nb_speaker, self.nb_sample_per_speaker, EMBEDDING) # (batch, embedding) -> (nb_speaker, nb_sample_per_speaker, embedding)

        support_set = x[:,:self.n_shot,:] # (nb_speaker, n_shot, embedding)
        query_set = x[:,self.n_shot:,:] # (nb_speaker, nb_sample_per_speaker - n_shot, embedding)
        query_set = query_set.reshape(self.nb_speaker * (self.nb_sample_per_speaker - self.n_shot), EMBEDDING) # (nb_speaker * (nb_sample_per_speaker - n_shot), embedding)

        prototype = torch.mean(support_set, dim=1) # (nb_speaker, embedding) or centroid

        # calculate distance
        logit = -1 * (torch.cdist(query_set, prototype)) # (nb_speaker * (nb_sample_per_speaker - n_shot), nb_speaker)

        # calculate loss
        label_metric = torch.arange(self.nb_speaker).repeat_interleave(self.nb_sample_per_speaker - self.n_shot).cuda() # (nb_speaker * (nb_sample_per_speaker - n_shot))
        nloss   = self.criterion(logit, label_metric)

        # calculate accuracy
        prec1   = accuracy(logit.detach(), label.detach(), topk=(1,))[0]

        return nloss, prec1