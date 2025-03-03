import os
import pickle
import numpy as np
import torch
from Backbones.model_factory import get_model
from Backbones.utils import evaluate, NodeLevelDataset, evaluate_batch
from training.utils import mkdir_if_missing
from dataset.utils import semi_task_manager
import importlib
import copy
import dgl
from tqdm import tqdm
import collections

def get_pipeline(args):
    if args.minibatch:
        if args.ILmode == 'classIL':
            return pipeline_class_IL_no_inter_edge_minibatch
        elif args.ILmode == 'taskIL':
            return pipeline_task_IL_no_inter_edge_minibatch
    else:
        if args.ILmode == 'classIL':
            return pipeline_class_IL_no_inter_edge
        elif args.ILmode == 'taskIL':
            return pipeline_task_IL_no_inter_edge


def data_prepare(args):
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    n_cls_so_far = 0
    # check whether the preprocessed data exist and can be loaded
    str_int_tsk = 'inter_tsk_edge' if args.inter_task_edges else 'no_inter_tsk_edge'
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        try:
            if args.load_check:
                subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(
                    f'{args.data_path}/{str_int_tsk}/{args.dataset}_{task_cls}.pkl', 'rb'))
            else:
                if f'{args.dataset}_{task_cls}.pkl' not in os.listdir(f'{args.data_path}/{str_int_tsk}'):
                    subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(
                        f'{args.data_path}/{str_int_tsk}/{args.dataset}_{task_cls}.pkl', 'rb'))
        except:
            print(f'preparing data for task {task}')
            mkdir_if_missing(f'{args.data_path}/inter_tsk_edge')
            mkdir_if_missing(f'{args.data_path}/no_inter_tsk_edge')
            if args.inter_task_edges:
                cls_retain = []
                for clss in args.task_seq[0:task + 1]:
                    cls_retain.extend(clss)
                subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = dataset.get_graph(
                    tasks_to_retain=cls_retain)
                with open(f'{args.data_path}/inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'wb') as f:
                    pickle.dump([subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids]], f)
            else:
                subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = dataset.get_graph(tasks_to_retain=task_cls)
                with open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'wb') as f:
                    pickle.dump([subgraph, ids_per_cls, [train_ids, valid_ids, test_ids]], f)

def pipeline_task_IL_no_inter_edge(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()

    model = get_model(dataset, args).cuda(args.gpu) if valid else None
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    data_prepare(args)
    n_cls_so_far = 0
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(
                f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))

        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)

        pbar =  tqdm(range(epochs))
        for epoch in pbar:
            pbar.set_description("Training task %s" %str(task))
            if args.method == 'lwf':
                life_model_ins.observe_task_IL(args, subgraph, features, labels, task, prev_model, train_ids,
                                               ids_per_cls, dataset)
            else:
                life_model_ins.observe_task_IL(args, subgraph, features, labels, task, train_ids, ids_per_cls, dataset)
        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        acc_mean = []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(
                    f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl','rb'))
            test_ids = valid_ids_ if valid else test_ids_
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            label_offset1, label_offset2 = task_manager.get_label_offset(t - 1)[1], task_manager.get_label_offset(t)[1]
            labels = labels - label_offset1
            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)
        prev_model = copy.deepcopy(model).cuda(args.gpu)

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_class_IL_no_inter_edge(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far+=len(task_cls)
        subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(
            open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))
        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)
        label_offset1, label_offset2 = task_manager.get_label_offset(task)

        pbar =  tqdm(range(epochs))
        for epoch in pbar:
            pbar.set_description("Training task %s" %str(task))
            if args.method == 'lwf':
                life_model_ins.observe(args, subgraph, features, labels, task, prev_model, train_ids, ids_per_cls, dataset)
            else:
                life_model_ins.observe(args, subgraph, features, labels, task, train_ids, ids_per_cls, dataset)
                torch.cuda.empty_cache()

        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        acc_mean = []

        for t in range(task+1):
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(
                f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2, cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, args.n_cls, cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)

            acc_matrix[task][t] = round(acc*100,2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc*100:.2f}|", end="")

        accs = acc_mean[:task+1]
        meana = round(np.mean(accs)*100,2)
        meanas.append(meana)
        acc_mean = round(np.mean(acc_mean)*100,2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)
        prev_model = copy.deepcopy(model).cuda()

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks-1):
        b = acc_matrix[args.n_tasks-1][t]-acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward),2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_task_IL_no_inter_edge_minibatch(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()

    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(
            open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl',
                 'rb'))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)

        dataloader = dgl.dataloading.NodeDataLoader(subgraph, train_ids, args.nb_sampler,
                                                    batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                    drop_last=False)

        pbar =  tqdm(range(epochs))
        for epoch in pbar:
            pbar.set_description("Training task %s" %str(task))
            if args.method == 'lwf':
                life_model_ins.observe_task_IL_batch(args, subgraph, dataloader, features, labels, task, prev_model, train_ids, ids_per_cls,
                                       dataset)
            else:
                life_model_ins.observe_task_IL_batch(args, subgraph, dataloader, features, labels, task, train_ids, ids_per_cls, dataset)
                torch.cuda.empty_cache()  # tracemalloc.stop()


        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        acc_mean = []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(
                f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            label_offset1, label_offset2 = task_manager.get_label_offset(t - 1)[1], task_manager.get_label_offset(t)[1]
            acc = evaluate_batch(args,model, subgraph, features, labels-label_offset1, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)
        prev_model = copy.deepcopy(model).cuda()

    print('AP: ', acc_mean)
    backward = []
    forward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)

    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_class_IL_no_inter_edge_minibatch(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()

    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0


    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(
            open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl','rb'))
        num_nodes = 0
        for ids in ids_per_cls:
            num_nodes+=len(ids)
        print ("Task: %s, Lables: %s, Node number: %s" %(str(task),str(task_cls), str(len(train_ids))))

        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)

        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]

        dataloader = dgl.dataloading.DataLoader(subgraph, train_ids, args.nb_sampler,
                                                    batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                    drop_last=False, use_uva=True)

        patience = 20
        best_loss = float('inf')
        epochs_no_improve = 0

        if valid:
            life_model_ins.opt.state = collections.defaultdict(dict)

        if args.show_pbar:
            pbar =  tqdm(range(epochs))
            pbar.set_description("Training task %s" %str(task))
        else:
            pbar = range(epochs)
        for epoch in pbar:
            loss = life_model_ins.observe_class_IL_batch(args, subgraph, dataloader, features, labels, 
                                                  task, train_ids, ids_per_cls_train, dataset, epoch)
            if args.early_stopping:
                if loss < best_loss:
                    best_loss = loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        loss = life_model_ins.observe_class_IL_batch(args, subgraph, dataloader, features, labels, task, train_ids, ids_per_cls_train, dataset, epochs-1)
                        print("Early stopping triggered!")
                        break

        label_offset1, label_offset2 = task_manager.get_label_offset(task)
        # test
        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        model.eval()
            
        acc_mean = []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        if t>0:
            backward = []
            for t in range(task):
                b = acc_matrix[task][t] - acc_matrix[t][t]
                backward.append(round(b, 2))
            mean_backward = round(np.mean(backward), 2)
            print(f"|AF: {mean_backward}", end="")

        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)
        prev_model = copy.deepcopy(model).cuda()

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

