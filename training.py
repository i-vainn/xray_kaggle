from tqdm import tqdm
import torch
import wandb

from sklearn.metrics import roc_auc_score


LABELS = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']

def columnwise_rocauc(preds, gts, prefix):
    res = {}
    for name, pred, gt in zip(LABELS, preds.T, gts.T):
        try:
            score = roc_auc_score(gt, pred)
        except:
            score = 0
        res[prefix.format(name + ' ROC-AUC')] = score
    res[prefix.format('Avg ROC-AUC')] = sum(res.values()) / len(res)
    return res

def run_epoch(model, dataloader, criterion, device='cpu', optimizer=None, do_train=True, scheduler=None):
    loss_log, preds, gts = [], [], []
    prefix = ['Val', 'Train'][do_train] + ' - {}'
    model.train(do_train)

    for it, (x_batch, y_batch) in tqdm(enumerate(dataloader)):
        data = x_batch.to(device)
        target = y_batch.to(device)

        if do_train:
            optimizer.zero_grad()

        with torch.inference_mode(not do_train):
            output = model(data)
            loss = criterion(output, target).cpu()
            
        preds.append(output.detach().cpu())
        gts.append(y_batch.detach().cpu())
        loss_log.append(loss.item())

        if it % wandb.config.log_iters == 0 and do_train:
            logs = {
                prefix.format('loss'): loss.item(),
                prefix.format('lr'): optimizer.param_groups[0]['lr'] if optimizer else None,
            }
            metric_stats = columnwise_rocauc(torch.cat(preds), torch.cat(gts), prefix)
            logs.update(metric_stats)
            wandb.log(logs)

        if not do_train:
            continue
    
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

    return torch.cat(gts), torch.cat(preds), loss_log
    
def train(model, dataloaders, optimizer, criterion, n_epochs, device='cpu', scheduler=None):
    best_val_score = 0.8
    prefix = 'Full val - {}'

    for epoch in tqdm(range(n_epochs)):
        print("Epoch {0} of {1}".format(epoch, n_epochs))

        train_targets, train_preds, train_loss = run_epoch(
            model=model,
            dataloader=dataloaders['train'],
            criterion=criterion, 
            optimizer=optimizer,
            do_train=True,
            scheduler=scheduler,
            device=device,
        )

        val_targets, val_preds, val_loss = run_epoch(
            model=model,
            dataloader=dataloaders['val'],
            criterion=criterion, 
            optimizer=None,
            do_train=False,
            scheduler=None,
            device=device,
        )

        logs = {
            prefix.format('loss'): sum(val_loss) / len(val_loss),
        }
        val_stats = columnwise_rocauc(val_preds, val_targets, prefix)
        logs.update(val_stats)
        wandb.log(logs)

        if val_stats[prefix.format('Avg ROC-AUC')] > best_val_score:
            best_val_score = val_stats[prefix.format('Avg ROC-AUC')]
            path = 'checkpoints/{}-{}-{:.4f}_model.pt'.format(wandb.run.name, epoch, best_val_score)
            torch.save(model.state_dict(), path)

def inference(model, dataloader, device='cpu'):
    preds = []
    model.eval()
    for x_batch, _ in tqdm(dataloader):
        data = x_batch.to(device)
        with torch.inference_mode():
            output = model(data)
            preds.append(output.detach().cpu())
            
    return torch.cat(preds)
