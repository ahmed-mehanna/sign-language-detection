import torch
from abc import ABC, abstractmethod
import copy
from tqdm import tqdm


class ModelTrainer(ABC):
    def __init__(self, model, opt, loss_func, lr_scheduler=None):
        self.model = model
        self.optimizer = opt
        self.loss_func = loss_func
        self.lr_scheduler = lr_scheduler

    def train_val(self, train_dl, val_dl, start_epoch, end_epoch, path_to_weights, path_to_checkpoint, device, loss_history=None, metric_history=None, progress_bar=True):
        loss_history, metric_history, best_model_wts, best_opt_hyp_params, best_lr_scheduler_hyp_params, best_loss_hyp_params, best_epoch, best_loss = self.__initialization(loss_history, metric_history)

        end_epoch += 1
        for epoch in range(start_epoch, end_epoch):
            current_lr = self.__get_lr()
            print('Epoch {}/{}, current lr={}'.format(epoch, end_epoch - 1, current_lr))
            if progress_bar:
                print("Training")
            self.model.train()
            train_loss, train_metric = self.__loss_epoch(train_dl, device, progress_bar)
            loss_history["train"].append(train_loss)
            metric_history["train"].append(train_metric)
            if val_dl is not None:
                if progress_bar:
                    print("Validation")
                # with torch.no_grad():
                self.model.eval()
                val_loss, val_metric = self.__loss_epoch(val_dl, device, progress_bar)

                loss_history["val"].append(val_loss)
                metric_history["val"].append(val_metric)
            else:
                val_loss = train_loss

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                self.__save_epoch(epoch, best_epoch, end_epoch - 1, train_dl.batch_size, loss_history, metric_history, path_to_weights)
                print("Copied best model weights!")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                if current_lr != self.__get_lr():
                    print("Loading best model weights!")
                    self.model.load_state_dict(best_model_wts)

            print("train loss: %.6f, train_accuracy: %.2f, val_loss: %.6f, val_accuracy: %.2f" %(train_loss, 100*train_metric, val_loss, 100*val_metric))        
            self.__save_epoch(epoch, best_epoch, end_epoch - 1, train_dl.batch_size, loss_history, metric_history, path_to_checkpoint)
            print("Checkpoint saved")

            print("-"*10)
        
        self.model.load_state_dict(best_model_wts)
        return self.model, loss_history, metric_history

    def test(self, data_dl, device, progress_bar=True):
        self.model.eval()
        loss, metric = self.__loss_epoch(data_dl, device, progress_bar)
        print("loss: %.6f, accuracy: %.2f" %(loss, 100*metric))
        return loss, metric

    def test2(self, data_dl):
        self.model.eval()
        correct = 0
        for data in data_dl:
            out = self.model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
        acc = correct / len(data_dl.dataset)
        return acc

    def __initialization(self, loss_history, metric_history):
        if loss_history is None:
            loss_history = {
                "train": [],
                "val": []
            }
        if metric_history is None:
            metric_history = {
                "train": [],
                "val": []
            }
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_opt_hyp_params = copy.deepcopy(self.optimizer.state_dict())
        if self.lr_scheduler is not None:
            best_lr_scheduler_hyp_params = copy.deepcopy(self.lr_scheduler.state_dict())
        else:
            best_lr_scheduler_hyp_params = None
        best_loss_hyp_params = copy.deepcopy(self.loss_func.state_dict())
        best_epoch = 0
        best_loss = float("inf")
        return loss_history, metric_history, best_model_wts, best_opt_hyp_params, best_lr_scheduler_hyp_params, best_loss_hyp_params, best_epoch, best_loss
    
    def __save_epoch(self, epoch, best_epoch, end_epoch, batch_size, loss_history, metric_history, path):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_opt_hyp_params = copy.deepcopy(self.optimizer.state_dict())
        if self.lr_scheduler is not None:
            best_lr_scheduler_hyp_params = copy.deepcopy(self.lr_scheduler.state_dict())
        else:
            best_lr_scheduler_hyp_params = None
        best_loss_hyp_params = copy.deepcopy(self.loss_func.state_dict())
        checkpoint = {
            "epochs": end_epoch,
            "last_epoch": epoch,
            "best_epoch": epoch,
            "batch_size": batch_size,
            "model_state_dict": best_model_wts,
            "optimizer_state_dict": best_opt_hyp_params,
            "lr_scheduler_state_dict": best_lr_scheduler_hyp_params,
            "loss_state_dict": best_loss_hyp_params,
            "hist": {
                "loss": loss_history,
                "metric": metric_history
            }
        }
        torch.save(checkpoint, path)
        torch.save(best_model_wts, path.split(".")[0]+".pth")

    def __loss_epoch(self, data_dl, device, progress_bar=True):
        def process(data, running_loss, running_metric):
            data.to(device, non_blocking=True)
            for param in params_name:
                params.append(data[param])
            output = self.model(*params)
            params.clear()
            loss_b = self.__loss_batch(output, data.y)
            with torch.no_grad():
                metric_b = self.__metrics_batch(output, data.y)
            running_loss += loss_b
            if metric_b is not None:
                running_metric += metric_b
            del data
            return running_loss, running_metric
        
        running_loss = 0.0
        running_metric = 0.0
        len_data = len(data_dl.dataset)
        params_name = self._get_params_name()
        params = []
        if progress_bar:
            for data in tqdm(data_dl):
                running_loss, running_metric = process(data, running_loss, running_metric)
        else:
            for data in data_dl:
                running_loss, running_metric = process(data, running_loss, running_metric)
        loss = running_loss/float(len_data)
        metric = running_metric/float(len_data)
        return loss, metric

    @abstractmethod
    def _get_params_name(self):
        pass
    
    def __loss_batch(self, output, target):
        loss = self.loss_func(output, target)
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()
    
    def __metrics_batch(self, outputs, targets):
        corrects = 0
        outputs = outputs.argmax(dim=1)
        corrects += int((outputs == targets).sum())
        return corrects

    def __get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

  


class GraphTrainer(ModelTrainer):
    def __init__(self, model, opt, loss_func, lr_scheduler):
        super(GraphTrainer, self).__init__(model, opt, loss_func, lr_scheduler)

    def _get_params_name(self):
        return ["x", "edge_index", "batch"]