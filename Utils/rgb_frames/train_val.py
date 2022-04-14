import copy
import torch
from tqdm import tqdm
import gc
from matplotlib import pyplot as plt

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def metrics_batch(outputs, targets):
    # pred = output.argmax(dim=1, keepdim=True)
    # corrects = pred.eq(target.view_as(pred)).sum().item()
    # return corrects
    
    corrects = 0
    for output, target in zip(outputs, targets):
        pred = output.argmax(keepdim=True)
        y = target.argmax(keepdim=True)
        # print(f"output: {output}, pred: {pred}, target: {target}, y: {y}")
        corrects += pred.eq(y).sum().item()
    return corrects

def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    with torch.no_grad():
        metric_b = metrics_batch(output, target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b

def loss_epoch(model, loss_func, dataset_dl, device, opt=None):
    running_loss=0.0
    running_metric=0.0
    len_data = len(dataset_dl.dataset)
    for xb, yb in tqdm(dataset_dl):
        xb, yb = xb.to(device), yb.to(device)
        # model = model.to(device)
        output=model(xb)
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        running_loss += loss_b
        
        if metric_b is not None:
            running_metric += metric_b
        
        del xb
        del yb
        gc.collect()

    loss = running_loss/float(len_data)
    metric = running_metric/float(len_data)
    return loss, metric

def train_val(model, loss_func, opt, train_dl, val_dl, lr_scheduler, batch_size, start_epoch, end_epoch, path_to_weights, path_to_checkpoint, device, loss_history=None, metric_history=None):   
    if loss_history is None:
        loss_history={
            "train": [],
            "val": [],
        }
    if metric_history is None:
        metric_history={
            "train": [],
            "val": [],
        }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_opt_hyp_params = copy.deepcopy(opt.state_dict())
    best_lr_scheduler_hyp_params = copy.deepcopy(lr_scheduler.state_dict())
    best_loss_hyp_params = copy.deepcopy(loss_func.state_dict())
    best_epoch = 0

    best_loss=float('inf')

    model = model.to(device)
    
    for epoch in range(start_epoch, end_epoch):
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, end_epoch - 1, current_lr))
        print("Training")
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, device, opt)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        print("Validation")
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, device)
        
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            best_opt_hyp_params = copy.deepcopy(opt.state_dict())
            best_lr_scheduler_hyp_params = copy.deepcopy(lr_scheduler.state_dict())
            best_loss_hyp_params = copy.deepcopy(loss_func.state_dict())
            best_epoch = epoch
            checkpoint = {
                "epochs": end_epoch,
                "last_epoch": epoch,
                "best_epoch": best_epoch,
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
            torch.save(checkpoint, path_to_weights)
            print("Copied best model weights!")
        
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts)


        print("train loss: %.6f, train_accuracy: %.2f, val_loss: %.6f, val_accuracy: %.2f" %(train_loss, 100*train_metric, val_loss, 100*val_metric))        
        checkpoint = {
                "epochs": end_epoch,
                "last_epoch": epoch,
                "batch_size": batch_size,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "loss_state_dict": loss_func.state_dict(),
                "hist": {
                    "loss": loss_history,
                    "metric": metric_history
                }
            }
        torch.save(checkpoint, path_to_checkpoint)
        print("Checkpoint saved")
        

        print("-"*10) 
    model.load_state_dict(best_model_wts)
        
    return model, loss_history, metric_history

def plot_loss(loss_hist, metric_hist):

    num_epochs= len(loss_hist["train"])

    plt.title("Train-Val Loss")
    plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
    plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show()

    plt.title("Train-Val Accuracy")
    plt.plot(range(1,num_epochs+1), metric_hist["train"],label="train")
    plt.plot(range(1,num_epochs+1), metric_hist["val"],label="val")
    plt.ylabel("Accuracy")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show()