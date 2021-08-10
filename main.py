import argparse
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import pickle
import random
from collections import Counter
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import *
from utils import *





def train_TC(csv_filename,expname,x_train_avail_label,y_train_avail_label,num_teacher,teachers,teachers_class,s_class,inputsize,hiddim,layers,ep,bs,lr,hiddimTTL,layersTTL,epTTL,bsTTL,lrTTL,x_train,x_test,y_train,y_test,save):
    
    # Train TTL
    row = ['Train TTL']; write_row(row, csv_filename, path="./output/", round=False)
    x_ttl = torch.FloatTensor(x_train_avail_label)
    y_ttl = prepare_y_TTL(num_teacher,y_train_avail_label,teachers_class)
    output_size=num_teacher; print_every=max(1,epTTL//10); clip=5;
    TTL_net = LSTM_multilabel(inputsize, output_size, hiddimTTL, layersTTL)
    TTL_net, total_loss, total_train_acc = train_TTL(x_ttl,y_ttl,bsTTL,TTL_net,lrTTL,epTTL,print_every,clip,csv_filename)
    plt_name = expname+'_TTL'
    plot_loss(plt_name+'_loss',total_loss,save_path='./output/plot/')
    
    # Train student
    row = ['Train STUDENT']; write_row(row, csv_filename, path="./output/", round=False)
    train_data=TensorDataset(x_train, torch.FloatTensor(y_train))
    train_loader=DataLoader(train_data, batch_size=bs, shuffle=True, drop_last=True)
    test_data=TensorDataset(x_test, torch.FloatTensor(y_test))
    test_loader=DataLoader(test_data, batch_size=bs, shuffle=False, drop_last=True)
    print_every=max(1,ep//10); clip=5; output_size=len(set(y_train)) # num_class_t1 + num_class_t2
    stu = LSTM(inputsize, output_size, hiddim, layers)
    params_1x = []; params_10x = []
    for name, param in stu.named_parameters():
        if 'fc' in name: params_10x.append(param)
        else: params_1x.append(param)
    optimizer = torch.optim.Adam([{'params': params_1x,         'lr': lr},
                                  {'params': params_10x,        'lr': lr*10}, ],
                                 lr=lr, weight_decay=1e-4)
    criterion_ce = nn.NLLLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_loss = []
    cur_epoch = 0
    best_score = 0.0

    # ===== Train Loop =====#
    while cur_epoch < ep:
        stu.train()
        ave_ep_loss, ep_loss, train_acc, test_acc, stu  = train_STU_ep(
                teachers_class=teachers_class,
                target_class=s_class,
                num_teacher=num_teacher,
                TTL_net=TTL_net,
                batch_size = bs,
                cur_epoch=cur_epoch,
                criterion_ce=criterion_ce,
                model=stu,
                teachers=teachers,
                optimizer=optimizer,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                print_every = print_every,
                clip = clip
                )
        total_loss.append(ep_loss)
        if test_acc > best_score:
            best_score = test_acc
            if save:
                filename = './output/'+expname+'.sav'
                pickle.dump(stu, open(filename, 'wb'))
        if cur_epoch % print_every == 0 or cur_epoch == ep-1 :
            row = [cur_epoch,'loss: {:.5f}'.format(ep_loss),"Test accuracy: {:.3f}".format(best_score)]
            write_row(row, 'output/'+csv_filename, path="./", round=False)

        cur_epoch += 1
        
    plt_name = expname+'_STUDENT'
    plot_loss(plt_name+'_loss',total_loss,save_path='./output/plot/')


def train_TTL(x_train,y_train,batch_size,model,lr,epochs,print_every,clip,csv_filename):
    train_data=TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_loss = []
    total_train_acc = []
    for t in range(epochs):
        epoch_loss = 0
        h = model.init_hidden(batch_size)
        for (inputs, labels) in train_loader:
            h = tuple([each.data for each in h])
            model.zero_grad()
            lstm_out, h, logits, sigmoid_out = model(inputs, h)
            loss = loss_fn(logits.squeeze(1)[:,-1], labels)
            loss.backward()
            epoch_loss += float(loss.data)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        train_acc = eval_LSTM_multilabel(model,train_loader,batch_size)
        total_loss.append(epoch_loss)
        total_train_acc.append(train_acc)
        if t % print_every == 0 or t == epochs-1 :
            row = [t,'loss: {:.5f}'.format(loss.item())]
            write_row(row, 'output/'+csv_filename, path="./", round=False)
    return model, total_loss, total_train_acc

def eval_LSTM_multilabel(net,train_loader,batch_size):
    eval_loss_fn = torch.nn.BCEWithLogitsLoss()
    h = net.init_hidden(batch_size)
    net.eval()
    for (inputs, labels) in train_loader:
        h = tuple([each.data for each in h])
        lstm_out, h, logits, softmax_out = net(inputs, h)
        eval_loss = eval_loss_fn(logits.squeeze(1)[:,-1], labels)
    return eval_loss



def train_STU_ep(teachers_class,target_class,num_teacher, TTL_net, batch_size, cur_epoch, criterion_ce, model, teachers, optimizer, train_loader, test_loader, device, print_every, clip, scheduler=None, vis=None, trace_name=None):
    softmax = nn.Softmax(dim=-1)
    t_pos_list = []
    t_h_list = []
    for n in range(num_teacher):
        t_h_list.append(teachers[n].init_hidden(batch_size))
        t_pos_list.append(n)

    epoch_loss = 0.0
    interval_loss = 0.0
    s_h = model.init_hidden(batch_size)
    conf_h = TTL_net.init_hidden(batch_size)
    
    for cur_step, (inputs, labels) in enumerate(train_loader):
        conf_lstm_out, conf_hidden, conf_logits, conf_softmax_out = TTL_net(torch.FloatTensor(inputs), conf_h)
        final_sigmoid = conf_softmax_out.squeeze(1)[:,-1]
        teacher_prob = softmax(final_sigmoid)
        t_conf_list = []
        for n in range(num_teacher):
            t_conf_list.append(teacher_prob[:,t_pos_list[n]])
        s_h = tuple([each.data for each in s_h])
        optimizer.zero_grad()
        model.zero_grad()
        with torch.no_grad():
            t_logits_w_conf_list = []
            t_logits_list = []
            t_softmax_out_list = []
            t_softmax_w_conf_list = []
            for n in range(num_teacher):
                t_lstm_out, t_h, t_logits, t_softmax_out = teachers[n](inputs, t_h_list[n])
                t_logits_w_conf = torch.transpose(t_conf_list[n].unsqueeze(0),1,0)*t_logits[:, -1]
                t_logits_w_conf_list.append(t_logits_w_conf)
                t_logits_list.append(t_logits[:, -1])
                t_softmax_out_list.append(t_softmax_out)
                t_softmax_w_conf = torch.transpose(t_conf_list[n].unsqueeze(0),1,0)*t_softmax_out[:, -1]
                t_softmax_w_conf_list.append(t_softmax_w_conf)
                
            t_logits = t_logits_list[0]
            t_logits_w_conf = t_logits_w_conf_list[0]
            t_softmax = t_softmax_out_list[0][:, -1]
            t_softmax_w_conf = t_softmax_w_conf_list[0]
            for n in range(1,num_teacher):
                t_logits = torch.cat((t_logits,t_logits_list[n]),dim=1) 
                t_logits_w_conf = torch.cat((t_logits_w_conf,t_logits_w_conf_list[n]),dim=1) 
                t_softmax = torch.cat((t_softmax,t_softmax_out_list[n][:, -1]),dim=1)
                t_softmax_w_conf = torch.cat((t_softmax_w_conf,t_softmax_w_conf_list[n]),dim=1)
                
        new_softmax = target_prob(batch_size,teachers_class,target_class,t_softmax_w_conf_list)
        hard_target = torch.argmax(new_softmax, dim=1)
        s_lstm_out, s_h, s_logits, s_softmax_out = model(inputs,s_h)
        loss = criterion_ce(torch.log(s_softmax_out[:, -1]), hard_target.type(torch.LongTensor)) #, None, False)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        np_loss = loss.detach().cpu().numpy()
        epoch_loss += np_loss
        interval_loss += np_loss

         # Accuracy
        train_acc = eval_LSTM(model,train_loader,batch_size)
        test_acc = eval_LSTM(model,test_loader,batch_size)
        avg_ep_loss = epoch_loss / len(train_loader)

    return avg_ep_loss, epoch_loss, train_acc, test_acc, model


def eval_LSTM(net,train_loader,batch_size):
    num_correct = 0
    h = net.init_hidden(batch_size)
    net.eval()
    for (inputs, labels) in train_loader:
        labels = labels.type(torch.LongTensor)
        h = tuple([each.data for each in h])
        lstm_out, h, logits, softmax_out = net(inputs, h)
        pred = torch.argmax(softmax_out.squeeze(1)[:,-1], dim=1)
        correct_tensor = pred.eq(labels.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)
    test_acc = num_correct/len(train_loader.dataset)
    return test_acc


def target_prob(batch_size,teachers_class,target_class,t_softmax_w_conf_list):
    num_teacher = len(teachers_class)
    target_size = [batch_size,len(target_class)]
    new_prob_mat = torch.zeros(target_size[0],target_size[1])
    for i in target_class:
        for t in range(num_teacher):
            if i in teachers_class[t]:
                if np.where(np.array(teachers_class[t])==i)[0].size > 0:
                    new_prob_class_i_t = t_softmax_w_conf_list[t][:,np.where(np.array(teachers_class[t])==i)[0][0]]
                    new_prob_mat[:,i] += new_prob_class_i_t
    return new_prob_mat





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    
    add_arg('-t_model',     nargs='+',  type=str,   required=True,  help='a list of paths of teacher models')
    add_arg('-t_numclass',  nargs='+',  type=int,   required=True,  help='the number of classes corresponding to t_model')
    add_arg('-t_class',     nargs='+',  type=int,   required=True,  help='a list of specialized classes of each teacher, concatenated in correspond to t_model, e.g., t1_class: 1 2 3 4 and t2_class: 3 4 5 6, then t_class: 1 2 3 4 3 4 5 6')
    add_arg('-s_class',     nargs='+',  type=int,   required=True,  help='a list of comprehensive classes of the student')
    add_arg('-data',                    type=str,   required=True,  help='the path of student training data file')
    add_arg('-expname',                 type=str,   required=True,  help='experiment name')
    add_arg('-lr',                      type=float, default=0.01,   help='Student: learning rate')
    add_arg('-ep',                      type=int,   default=200,    help='Student: total epochs for training')
    add_arg('-bs',                      type=int,   default=8,      help='Student: batch size')
    add_arg('-layers',                  type=int,   default=2,      help='Student: #layers')
    add_arg('-hiddim',                  type=int,   default=8,      help='Student: #hidden units')
    add_arg('-lrTTL',                   type=float, default=0.01,   help='TTL: learning rate')
    add_arg('-epTTL',                   type=int,   default=500,    help='TTL: total epochs for training')
    add_arg('-bsTTL',                   type=int,   default=8,      help='TTL: batch size')
    add_arg('-layersTTL',               type=int,   default=2,      help='TTL: #layers')
    add_arg('-hiddimTTL',               type=int,   default=8,      help='TTL: #hidden units')
    add_arg('-inputsize',               type=int,   default=1,      help='#features')
    add_arg('-seed',                    type=int,   default=0,      help='set seed for reproduction')
    add_arg('-plabel',                  type=float, default=0.02,      help='proportion of available labeled data (range = [0,1])')
    add_arg('--save',                   action='store_true',        help='boolean parameters, whether to save the student model')
    
    args = parser.parse_args()
    t_model = args.t_model
    t_numclass = args.t_numclass
    t_class = args.t_class 
    s_class = args.s_class
    data = args.data
    expname = args.expname
    lr = args.lr
    ep = args.ep
    bs = args.bs
    layers = args.layers
    hiddim = args.hiddim
    lrTTL = args.lrTTL
    epTTL = args.epTTL
    bsTTL = args.bsTTL
    layersTTL = args.layersTTL
    hiddimTTL = args.hiddimTTL
    inputsize = args.inputsize
    seed = args.seed
    save = args.save
    plabel = args.plabel
    num_teacher = len(t_model)
    
    
    # Map class
    labels_map = labelmap(s_class)
    t_class_mapped = list(map(labels_map.get, t_class))
    s_class = list(map(labels_map.get, s_class))
    teachers_class = []
    bg = 0
    for t in range(num_teacher):
        t_classset = t_class_mapped[bg:bg+t_numclass[t]]
        teachers_class.append(t_classset)
        bg += t_numclass[t]
            
    # Laod pretrained teachers
    teachers = [] 
    for t in range(num_teacher):
        filename_t = t_model[t]
        net_t = pickle.load(open(filename_t, 'rb'))
        teachers.append(net_t)

    
    # Prepare data
    X, Y = read_data(data)
    x = torch.FloatTensor(X).unsqueeze(2)
    y = np.array(list(map(labels_map.get, Y)))
    n_splits = 1; test_size = 0.2
    x_train, x_test, y_train, y_test = stratified_sampling(n_splits,test_size,x,y,seed)
    
    # -- Split labeled data
    n_splits = 1; avail_label = plabel
    x_train_no_label, x_train_avail_label, y_train_no_label, y_train_avail_label = stratified_sampling(n_splits,avail_label,x_train,y_train,seed)
    
    
    csv_filename = expname
    row = ['Experiment Name', expname]
    write_row(row, csv_filename, path="./output/", round=False)
    
    train_TC(csv_filename,expname,x_train_avail_label,y_train_avail_label,num_teacher,teachers,teachers_class,s_class,inputsize,hiddim,layers,ep,bs,lr,hiddimTTL,layersTTL,epTTL,bsTTL,lrTTL,x_train,x_test,y_train,y_test,save)
    
    

    
    
    







