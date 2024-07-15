from transformers import AutoFeatureExtractor, CvtForImageClassification
from torch.utils.data import Dataset, DataLoader
from string import digits
from sklearn.model_selection import train_test_split
from torchmetrics.functional.classification import auroc, accuracy

import torchvision
import torch
import torch.nn as nn
import pandas as pd
import os
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import Config

torch.set_float32_matmul_precision('medium')


class CVT_Fairface(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.config['model_name'])
        self.model = CvtForImageClassification.from_pretrained(config['model_name']).to(self.config['device'])
        self.new_classifier = nn.Linear(384, self.config['n_labels']).to(self.config['device'])
        torch.nn.init.xavier_uniform_(self.new_classifier.weight)
        self.model.classifier = self.new_classifier
        self.softmax = nn.Softmax(dim = 1)
        self.loss = nn.CrossEntropyLoss()
        self.training_step_outputs = []
        self.training_auroc = []
        self.training_acc = []
        self.validation_step_outputs = []
        self.validation_auroc = []
        self.validation_acc = []
        self.test_step_outputs = []
        self.test_auroc = []
        self.test_acc = []
        self.tloss = []
        self.tauroc = []
        self.tacc = []
        self.vloss = []
        self.vauroc = []
        self.vacc = []
           
    def forward(self, x, labels = None):
        features = x.to(self.config['device'])
        out = self.model(features)
        return out.logits
    
    def training_step(self, batch, batch_index):
        loss, out, y = self._common_step(batch, batch_index)
        pred = self.softmax(out)
        t_auroc = auroc(pred, y, task = 'multiclass', num_classes = self.config['n_labels'])
        t_acc = accuracy(pred, y, task = 'multiclass', num_classes = self.config['n_labels'])
        self.training_step_outputs.append(loss)
        self.training_auroc.append(t_auroc)
        self.training_acc.append(t_acc)
        self.log("Training Accuracy", t_acc, prog_bar = True, logger = True)
        return loss
    
    def on_train_epoch_end(self):
        epoch_mean = torch.stack(self.training_step_outputs).mean()
        epoch_auroc = torch.stack(self.training_auroc).mean()
        epoch_acc = torch.stack(self.training_acc).mean()
        self.tloss.append(float(epoch_mean.detach().cpu().numpy()))
        self.tauroc.append(float(epoch_auroc.detach().cpu().numpy()))
        self.tacc.append(float(epoch_acc.detach().cpu().numpy()))
        self.training_step_outputs.clear()
        self.training_auroc.clear()
        self.training_acc.clear()
    
    def validation_step(self, batch, batch_index):
        loss, out, y = self._common_step(batch, batch_index)
        pred = self.softmax(out)
        v_auroc = auroc(pred, y, task = 'multiclass', num_classes = self.config['n_labels'])
        v_acc = accuracy(pred, y, task = 'multiclass', num_classes = self.config['n_labels'])
        self.validation_step_outputs.append(loss)
        self.validation_auroc.append(v_auroc)
        self.validation_acc.append(v_acc)
        self.log("Validation Accuracy", v_acc, prog_bar = True, logger = True)
        return loss
    
    def on_validation_epoch_end(self):
        epoch_mean = torch.stack(self.validation_step_outputs).mean()
        epoch_auroc = torch.stack(self.validation_auroc).mean()
        epoch_acc = torch.stack(self.validation_acc).mean()
        self.vloss.append(float(epoch_mean.detach().cpu().numpy()))
        self.vauroc.append(float(epoch_auroc.detach().cpu().numpy()))
        self.vacc.append(float(epoch_acc.detach().cpu().numpy()))
        self.validation_step_outputs.clear()
        self.validation_auroc.clear()
        self.validation_acc.clear()
    
    def test_step(self, batch, batch_index):
        loss, out, y = self._common_step(batch, batch_index)
        pred = self.softmax(out)
        t_auroc = auroc(pred, y, task = 'multiclass', num_classes = self.config['n_labels'])
        t_acc = accuracy(pred, y, task = 'multiclass', num_classes = self.config['n_labels'])
        self.test_step_outputs.append(loss)
        self.test_auroc.append(t_auroc)
        self.test_acc.append(t_acc)
        self.log("Test Accuracy", t_acc, prog_bar = True, logger = True)
        return loss

    def predict_step(self, batch, batch_index):
        loss, out, y = self._common_step(batch, batch_index)
        return loss
    
    def _common_step(self, batch, batch_index):
        x, y = batch
        x = x.type(torch.cuda.FloatTensor)
        y = y.type(torch.LongTensor)
        y = y.to(self.config['device'])
        out = self.forward(x)
        loss = self.loss(out, y)
        return loss, out, y
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.config['lr'])
        return [optimizer]
    
    def plot_loss(self):
        self.vloss.pop()
        plt.plot(self.tloss, label = 'Training')
        plt.plot(self.vloss, label = 'Validation')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
    def plot_auroc(self):
        self.vauroc.pop()
        plt.plot(self.tauroc, label = 'Training')
        plt.plot(self.vauroc, label = 'Validation')
        plt.title('AUROC')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
    def plot_accuracy(self):
        self.vacc.pop()
        plt.plot(self.tacc, label = 'Training')
        plt.plot(self.vacc, label = 'Validation')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
