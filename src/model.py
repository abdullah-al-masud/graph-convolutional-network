"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

import torch
import torch_geometric as pyg
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from msdlib.msd import get_time_estimation
import numpy as np
import os
import time
from msdlib.mlutils import modeling
from msdlib.mlutils.utils import store_models
import pdb


class modified_Tmodel(modeling.torchModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def fit(self, data=None, label=None, val_data=None, val_label=None, validation_ratio=.15, evaluate=True, figsize=(18, 4), train_loader=None, val_loader=None):
        """
        scikit like wrapper for training DNN pytorch model
        Inputs:
            :data: input train data, must be torch tensor, numpy ndarray or pandas DataFrame/Series. Default value in None
            :label: supervised labels for data, must be torch tensor, numpy ndarray or pandas DataFrame/Series. Default value in None
            :val_data: validation data, must be torch tensor, numpy ndarray or pandas DataFrame/Series, default is None
            :val_label: supervised labels for val_data, must be torch tensor, numpy ndarray or pandas DataFrame/Series, default is None
            :validation_ratio: ratio of 'data' that will be used for validation during training. It will be used only when val_data or val_label or both are None. Default is 0.15
            :evaluate: bool, whether to evaluate model performance after training ends or not. evaluate performance if set True. Default is True.
            :figsize: tuple of (width, height) of the figure, size of the figure for loss curve plot and evaluation plots. Default is (18, 4)
            :train_loader: torch.utils.data.DataLaoder instance for handling training data set. Default is None
            :val_loader: torch.utils.data.DataLaoder instance for handling validation data set. Default is None
        Outputs:
            :self: torchModel object, returns the self object variable just like scikit-learn model objects
        """

        train_loader, val_loader = self.prepare_data_loader(
            data, label, val_data, val_label, validation_ratio, train_loader, val_loader)

        # running through epoch
        loss_curves = [[], []]
        val_loss = torch.tensor(np.nan)
        t1 = time.time()
        self.set_parallel()
        total_batch = len(train_loader)
        for ep in range(self.epoch):
            tr_mean_loss = []
            self.model.train()
            for i, (batch_data) in enumerate(train_loader):

                batch_label = batch_data.y
                # loss calculation
                self.model.zero_grad()
                label_hat = self.model(batch_data.to(device=self.device))
                tr_loss = self.loss_func(label_hat.squeeze(), batch_label.to(device=self.device).squeeze())

                # back-propagation
                tr_loss.backward()
                # model parameter update
                self.optimizer.step()

                # stacking and printing losses
                tr_mean_loss.append(tr_loss.item())
                time_string = get_time_estimation(
                    time_st=t1, current_ep=ep, current_batch=i, total_ep=self.epoch, total_batch=total_batch)
                print('\repoch : %04d/%04d, batch : %03d, train_loss : %.4f, validation_loss : %.4f,  %s'
                      % (ep + 1, self.epoch, i + 1, tr_loss.item(), val_loss.item(), time_string)+' '*20, end='', flush=True)

            # loss scheduler step
            self.scheduler.step()
            # storing losses
            tr_mean_loss = np.mean(tr_mean_loss)
            loss_curves[0].append(tr_mean_loss)

            if len(val_loader) > 0:
                # run evaluation to get validation score
                out, _val_label = self.predict(val_loader, return_label=True)
                val_loss = self.loss_func(out.squeeze(), _val_label.squeeze())
                # storing losses
                loss_curves[1].append(val_loss.item())
            else:
                loss_curves[1].append(np.nan)

			# tensorboard parameter collection
            self.add_tb_params(ep, tr_mean_loss, val_loss, _val_label.squeeze(), out, batch_data)

            # storing model weights after each interval
            if self.interval is not None:
                if (ep + 1) % self.interval == 0:
                    if self.parallel:
                        model_dict = {"%s_epoch-%d"%(self.model_name, ep+1): self.model.module}
                    else:
                        model_dict = {"%s_epoch-%d"%(self.model_name, ep+1): self.model}
                    store_models(model_dict, self.savepath)
	
        print('...training complete !!')
        losses = pd.DataFrame(loss_curves, index=['train_loss', 'validation_loss'], columns=np.arange(
            1, self.epoch + 1)).T.rolling(self.loss_roll_period).mean()

        # plotting loss curve
        if self.plot_loss and self.epoch > 1:
            ylim_upper = losses.quantile(self.quant_perc).max()
            ylim_lower = losses.min().min()
            fig, ax = plt.subplots(figsize=(25, 4))
            losses.plot(ax=ax, color=['darkcyan', 'crimson'])
            ax.set_ylim(ylim_lower, ylim_upper)
            fig.suptitle('Learning curves', y=1,
                         fontsize=15, fontweight='bold')
            fig.tight_layout()
            if self.savepath is not None:
                os.makedirs(self.savepath, exist_ok=True)
                fig.savefig('%s/Learning_curves.png' %
                            (self.savepath), bbox_inches='tight')
                plt.close()
            else:
                plt.show()

        self.relieve_parallel()

        if self.interval is not None:
            model_dict = {self.model_name: self.model}
            store_models(model_dict, self.savepath)

        # model training evaluation
        if evaluate:
            self.evaluate([train_loader.dataset.data, val_loader.dataset.data], 
                          [train_loader.dataset.label, val_loader.dataset.label], 
                          set_names=['Train_set (from training data)', 'Validation_set (from training data)'], 
                          figsize=figsize, savepath=self.savepath)

        return self
    

    def predict(self, data, return_label=False):
        """
        A wrapper function that generates prediction from pytorch model
        Inputs:
            :data: input data to predict on, can be a torch tensor, numpy ndarray, pandas DataFrame or even torch DataLoader object
            :return_label: bool, whether to return label data if 'data' is a pytorch DataLoader object. Default is False
        Outputs:
            :preds: torch Tensor, predicted values against the inserted data
            :labels: torch Tensor, only found if 'data' is pytorch DataLoader and return_label is True.
        """

        # evaluation mode set up
        self.model.eval()

        if isinstance(data, torch.utils.data.dataloader.DataLoader):
            with torch.no_grad():
                preds = []
                labels = []
                for i, (batch) in enumerate(data):
                    label = batch.y
                    pred = self.model(batch.to(device=self.device))
                    preds.append(pred.detach())
                    if return_label:
                        labels.append(label)
                preds = torch.cat(preds)
                if return_label:
                    labels = torch.cat(labels).to(device=self.device)
                    if self.model_type.lower() == 'multi-classifier':
                        labels = labels.to(dtype=torch.long)
                    else:
                        labels = labels.to(dtype=self.dtype)
                    return preds, labels
                else:
                    return preds
        
        else:
            # checking data type
            if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
                data = data.values
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)

            # estimating number of mini-batch
            n_batch = data.shape[0] // self.batch_size + \
                int(bool(data.shape[0] % self.batch_size))

            with torch.no_grad():
                # generates prediction
                preds = []
                for i in range(n_batch):
                    if i != n_batch - 1:
                        pred = self.model(
                            data[i * self.batch_size: (i + 1) * self.batch_size].to(device=self.device))
                    else:
                        pred = self.model(
                            data[i * self.batch_size:].to(device=self.device))
                    preds.append(pred.detach())
                preds = torch.cat(preds)
            
            return preds



class Regressor(torch.nn.Module):

    def __init__(self, node_features, edge_features):
        super(Regressor, self).__init__()

        self.build_model(node_features, edge_features)
    

    def build_model(self, node_features, edge_features, units=32):
        self.node_layer1 = pyg.nn.GATConv(node_features, units, egde_dim=edge_features)
        self.edge_layer1 = torch.nn.Linear(units * 2 + edge_features, units)
        self.node_layer2 = pyg.nn.GATConv(units, units, egde_dim=units)
        self.pool = pyg.nn.global_mean_pool
        self.out = torch.nn.Linear(units, 1)

    def forward(self, data):

        node_features, edge_index, edge_features, batch = data.x, data.edge_index, data.edge_attr, data.batch

        node_features = self.node_layer1(x=node_features, edge_index=edge_index, edge_attr=edge_features)
        edge_features = self.edge_layer1(torch.cat([node_features[edge_index[0]], node_features[edge_index[1]], edge_features], axis=1))
        node_features = self.node_layer2(x=node_features, edge_index=edge_index, edge_attr=edge_features)
        graph_features = self.pool(node_features, batch)
        prediction = self.out(graph_features)

        return prediction

class Classifier(torch.nn.Module):

    def __init__(self, node_features, edge_features, classes):
        super(Classifier, self).__init__()

        self.build_model(node_features, edge_features, classes)
    

    def build_model(self, node_features, edge_features, classes):
        self.node_layer1 = pyg.nn.GATConv(node_features, 32, egde_dim=edge_features)
        self.edge_layer1 = torch.nn.Linear(node_features * 2 + edge_features, 32)
        self.node_layer2 = pyg.nn.GATConv(32, 32, egde_dim=32)
        self.edge_layer2 = torch.nn.Linear(32 * 2 + 32, 32)
        self.classifier = torch.nn.Linear(32 * 2 + 32, classes)

    def forward(self, data):

        node_features, edge_index, edge_features = data.x, data.edge_index, data.edge_attr

        node_features = self.node_layer1(x=node_features, edge_index=edge_index, edge_attr=edge_features)
        edge_features = self.edge_layer1(torch.cat([node_features[edge_index[0]], node_features[edge_index[1]], edge_features], axis=1))
        node_features = self.node_layer2(x=node_features, edge_index=edge_index, edge_attr=edge_features)
        edge_features = self.edge_layer2(torch.cat([node_features[edge_index[0]], node_features[edge_index[1]], edge_features], axis=1))
        prediction = self.classifier(edge_features)

        return prediction

        
