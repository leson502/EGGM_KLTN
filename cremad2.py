import comet_ml
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.cremadv2 import CREMADModel, ClassifierGuided
from sklearn.metrics import confusion_matrix, classification_report
from models.moe import cv_squared
from src.eval_metrics import eval_cls, cal_cos
from tqdm import tqdm
from tensorboardX import SummaryWriter

steps = 0

def initiate(hyp_params, train_loader, test_loader):
    model = CREMADModel()

    if hyp_params.modulation != 'none':
        classifier = ClassifierGuided(2, 768)
        cls_optimizer = getattr(optim, hyp_params.optim)(classifier.parameters(), lr=hyp_params.cls_lr)
    else:
        classifier, cls_optimizer = None, None

    if hyp_params.use_cuda:
        model = model.cuda()
        if hyp_params.modulation != 'none':
            classifier = classifier.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    
    criterion = getattr(nn, hyp_params.criterion)()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    writer = SummaryWriter(comment='CREMAD', comet_config={'project_name': 'CREMAD', "disabled": True})
    settings = {'model': model, 'optimizer': optimizer, 'criterion': criterion, 'scheduler': scheduler,
                'classifier': classifier, 'cls_optimizer': cls_optimizer, 'writer': writer}
    return train_model(settings, hyp_params, train_loader, test_loader)




def train_model(settings, hyp_params, train_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']
    classifier = settings['classifier']
    cls_optimizer = settings['cls_optimizer']
    writer = settings['writer']
    acc1 = [0] * hyp_params.num_mod
    l_gm = None
    coeff = None
    

    def train(model, classifier, optimizer, cls_optimizer, criterion):
        nonlocal acc1, l_gm, coeff
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        global steps
        gate_list = {i: [] for i in range(hyp_params.num_mod)}

        def getting_gate(module, input, output):
                inp, x = input
                gates, load = output
                select = (gates > 0).long().sum(0)
                gate_list[x].append(select)

        for batch in tqdm(train_loader):
            audio, vision, batch_Y = batch
            eval_attr = batch_Y.squeeze(-1)  # if num of labels is 1
            model.zero_grad()
            classifier.init_classifier(model.classifier)

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    audio, vision, eval_attr = audio.cuda(), vision.cuda(), eval_attr.cuda()
                    eval_attr = eval_attr.long()

            handles = []
            for module in model.classifier.gatting_network:
                handles.append(module.register_forward_hook(getting_gate))
            
            batch_size = audio.size(0)
            net = nn.DataParallel(model) if batch_size > 10 else model
            preds, hs, gate_load = net(audio, vision)

            for handle in handles:
                handle.remove()
   
            raw_loss = criterion(preds, eval_attr) 
            if hyp_params.modulation == 'cggm':
                if l_gm is not None:
                    raw_loss += hyp_params.lamda * l_gm
                if coeff is not None:
                    for i in range(hyp_params.num_mod):
                        gate_load[0][i] = gate_load[0][i] / coeff[i]
                        gate_load[1][i] = gate_load[1][i] / coeff[i]

                    
            gate = sum(gate_load[0])
            load = sum(gate_load[1])

            router_loss = hyp_params.beta * (cv_squared(gate) + cv_squared(load))
            raw_loss += router_loss
            raw_loss.backward()
            
            if hyp_params.modulation == 'cggm':
                cls_optimizer.zero_grad()
                net2 = nn.DataParallel(classifier) if batch_size > 10 else classifier
                cls_res = net2(hs)

                for name, para in net.named_parameters():
                    if 'out_layer.weight' in name:
                        fusion_grad = para
                
                cls_loss = criterion(cls_res[0], eval_attr)
                for i in range(0, hyp_params.num_mod):
                    uni_cls_loss = criterion(cls_res[i], eval_attr)
                    cls_loss += uni_cls_loss
                    writer.add_scalar(f'loss/cls_{i}', uni_cls_loss.item(), steps)

                cls_loss = cls_loss / hyp_params.num_mod
                cls_loss.backward()

                cls_optimizer.step()

                cls_grad = []
                for name, para in net2.named_parameters():
                    if 'out_layer.weight' in name:
                        cls_grad.append(para)                

                
                llist = cal_cos(cls_grad, fusion_grad)
                
                acc2 = classifier.cal_coeff(eval_attr, cls_res)
                diff = [max(acc2[i] - acc1[i], 0) for i in range(hyp_params.num_mod)]


                diff_sum = sum(diff) + 1e-8
                coeff = list()

                for d in diff:
                    coeff.append((diff_sum - d) / diff_sum)
                
                for i in range(hyp_params.num_mod):
                    writer.add_scalar(f'acc/cls_{i}', acc2[i], steps)
                    writer.add_scalar(f'coeff/cls_{i}', coeff[i], steps)
                
                for i in range(hyp_params.num_mod):
                    acc1[i] = acc2[i]

                l_gm = np.sum(np.abs(coeff)) - (coeff[0] * llist[0] + coeff[1] * llist[1])
                l_gm /= hyp_params.num_mod

                for i in range(hyp_params.num_mod):
                    for name, params in net.named_parameters():
                        if f'encoder_{i}' in name:
                            params.grad *= (coeff[i] * hyp_params.rou)
                
                steps += 1
            


            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += raw_loss.item() * batch_size

            writer.add_scalar('loss/train', raw_loss.item() / batch_size, steps)
        
        for i in range(hyp_params.num_mod):
                gate_list[i] = torch.cat(gate_list[i], dim=0).sum().item()
                # add bar plot for gate_list
                writer.add_scalar(f'gate/cls_{i}', gate_list[i], steps)

        writer.add_scalar('loss/train_epoch', epoch_loss / hyp_params.n_train, steps)

        return epoch_loss / hyp_params.n_train

    def evaluate(model, criterion):
        model.eval()
        loader = test_loader
        total_loss = 0.0

        results = []
        truths = []

        with torch.no_grad():
            for i_batch, batch in enumerate(loader):
                audio, vision, batch_Y = batch
                eval_attr = batch_Y.squeeze(dim=-1)  # if num of labels is 1

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        audio, vision, eval_attr = audio.cuda(), vision.cuda(), eval_attr.cuda()

                net = model
                preds, _, _ = net(audio, vision)

                total_loss += (criterion(preds, eval_attr)).item() 
                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_acc = 0
    best_f1 = 0
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train(model, classifier, optimizer, cls_optimizer, criterion)
        val_loss, val_res, val_truth = evaluate(model, criterion)
        
        acc, f1 = eval_cls(val_truth, val_res)

        end = time.time()
        duration = end - start
        scheduler.step(val_loss)  # Decay learning rate by validation loss

        print("-" * 50)
        print(
            'Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Acc {:5.4f} | F1 {:5.4f}'.format(epoch, duration, val_loss, acc, f1))
        print("-" * 50)

        if best_f1 < f1:
            best_acc = acc
            best_f1 = f1
            golds = val_truth.cpu().numpy()
            preds = np.argmax(val_res.cpu().numpy(), axis=1)
            
            cfm = confusion_matrix(golds, preds)
            report = classification_report(golds, preds, output_dict=True)

        writer.add_scalar('loss/valid', val_loss, steps)
        writer.add_scalar('acc/valid', acc, steps)

    writer.add_scalar('acc/best', best_acc, steps)
    writer.add_scalar('f1/best', best_f1, steps)
    writer.close()
    print("Accuracy: ", best_acc)
    print("F1: ", best_f1)
    print("Confusion Matrix: ")
    print(cfm)
    print("Classification Report: ")
    print(report)