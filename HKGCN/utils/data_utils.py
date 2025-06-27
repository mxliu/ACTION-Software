"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from scipy.linalg import fractional_matrix_power, inv
import torch
import manifolds
import pymetis as metis


def load_data(args, datapath):
    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
        data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats)

        # true labels
        labels_onehot = encode_onehot(data['labels'].numpy())
        train_mask = sample_mask(data['idx_train'], labels_onehot.shape[0])
        print("idx_train and true_labels: ", len(data['idx_train']), labels_onehot.shape[0])
        y_train = np.zeros(labels_onehot.shape)
        y_train[train_mask, :] = labels_onehot[train_mask, :]
        data['y_train'] = y_train

        if args.self_supervised != 'non':

            # load pre-trained pseudo labels
            data['idx_unlabel'] = list(set(range(data['features'].shape[0])) - set(data['idx_train']))
            #pseudo_label_path = './pseudo_label/nc_new/' + args.dataset + '_c4_' + args.clu_type + '.npy'
            pseudo_label_path = './pseudo_label/nc/' + args.dataset + '_' + args.clu_type + '.npy'
            data['pseudo_labels'] = np.load(pseudo_label_path)
            data['pseudo_adj'] = pseudo_adjacency(data)
            data['completed_adj'] = complete_adjacency(data['adj_train'], data['pseudo_adj'], args.split_seed)

            # damaged features
            idx = np.random.permutation(data['features'].shape[0])
            data['features_da'] = data['features'][idx, :]

            data['pseudo_adj_norm'], _ = process(data['pseudo_adj'], data['features'], args.normalize_adj, args.normalize_feats)
            data['completed_adj_norm'], _ = process(data['completed_adj'], data['features'], args.normalize_adj, args.normalize_feats)


    
    elif args.task == 'lp':
        data = load_data_lp(args.dataset, args.use_feats, datapath)
        adj = data['adj_train']
        masked_adj = mask_adjacency(adj, args.mask_prop, args.split_seed)

        adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
            masked_adj, args.val_prop, args.test_prop, args.split_seed
        )

        np.save('./pseudo_label/{}_adj_train_{}.npy'.format(args.dataset, args.mask_prop), adj_train)
            
        data['adj_train'] = adj_train
        data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
        data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
        data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false

        data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats)

        #fc = fc_prob(args, data)

        if args.self_supervised != 'non':

            pseudo_label_path = './pseudo_label/lp/' + args.dataset + '_link_' + args.clu_type + '.npy'
            data['pseudo_labels'] = np.load(pseudo_label_path)
            
            ### partition to compute (node) pseudo labels
            #data['pseudo_labels'] = partition(data['adj_train'], 50)

            ### completion to compute pseudo labels
            #data['pseudo_labels'] = completion(data['features'], 64)

            pseudo_adj = pseudo_adjacency(data)
            #pseudo_adj_train, pseudo_train_edges, pseudo_train_edges_false, pseudo_val_edges, pseudo_val_edges_false, pseudo_test_edges, pseudo_test_edges_false, pseudo_adj = mask_edges_pseudo(
            #    data, args.val_prop, args.test_prop, args.split_seed)

            data['pseudo_adj'] = pseudo_adj
            data['pseudo_pos_edges'], data['pseudo_neg_edges'] = mask_edges_train(pseudo_adj)
            #data['pseudo_adj_train'] = pseudo_adj_train
            #data['pseudo_train_edges'], data['pseudo_train_edges_false'] = pseudo_train_edges, pseudo_train_edges_false
            #data['pseudo_val_edges'], data['pseudo_val_edges_false'] = pseudo_val_edges, pseudo_val_edges_false
            #data['pseudo_test_edges'], data['pseudo_test_edges_false'] = pseudo_test_edges, pseudo_test_edges_false
            
            completed_adj = complete_adjacency(masked_adj, pseudo_adj, args.split_seed)
            data['completed_adj'] = completed_adj
            data['completed_pos_edges'], data['completed_neg_edges'] = mask_edges_train(completed_adj)

            idx = np.random.permutation(data['features'].shape[0])
            data['features_da'] = data['features'][idx, :]
            
            data['pseudo_adj_norm'], _ = process(data['pseudo_adj'], data['features'], args.normalize_adj, args.normalize_feats)
            data['completed_adj_norm'], _ = process(data['completed_adj'], data['features'], args.normalize_adj, args.normalize_feats)

    elif args.task == 'ad':
        data = load_data_lp(args.dataset, args.use_feats, datapath)
        #data['train_edges'], data['train_edges_false'] = mask_edges_train(data['adj_train'])
        #data['adj_train_norm'], data['features'] = process(
        #    data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats)

        ### anomaly data
        data_mat = sio.loadmat("./data/{}.mat".format(args.dataset))
        ano_label = data_mat['Label'] if ('Label' in data_mat) else data_mat['gnd']
        data['ano_label'] = np.squeeze(np.array(ano_label))
        attr = data_mat['Attributes'] if ('Attributes' in data_mat) else data_mat['X']
        network = data_mat['Network'] if ('Network' in data_mat) else data_mat['A']
        data['ano_adj'] = sp.csr_matrix(network)
        data['ano_features'] = sp.lil_matrix(attr)
        #data['ano_adj_norm'], data['ano_features'] = process(
        #   data['ano_adj'], data['ano_features'], args.normalize_adj, args.normalize_feats)

        ### (X_ano, A_ano)
        data['adj_train_norm'], data['features'] = process(
            data['ano_adj'], data['ano_features'], args.normalize_adj, args.normalize_feats)
        
        data['pos_adj'] = data['ano_adj'].todense()

        #data['train_edges'], data['train_edges_false'] = mask_edges_train(data['ano_adj'])
        adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
            data['ano_adj'], args.val_prop, args.test_prop, args.split_seed)
        data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
        data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
        data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false

        if args.self_supervised != 'non':
            pseudo_label_path = './pseudo_label/ad/' + args.dataset + '_ano_' + args.clu_type + '.npy'
            data['pseudo_labels'] = np.load(pseudo_label_path)
            data['pseudo_adj'] = pseudo_adjacency(data)
            data['pseudo_pos_edges'], data['pseudo_neg_edges'] = mask_edges_train(data['pseudo_adj'])
            data['pseudo_adj_norm'], _ = process(
                data['pseudo_adj'], data['features'], args.normalize_adj, args.normalize_feats)

            # damaged features
            idx = np.random.permutation(data['features'].shape[0])
            data['features_da'] = data['features'][idx, :]

    if args.self_supervised == 'contrast':
        a = data['adj_train'].toarray()
        a = a + np.eye(a.shape[0])  # A^ = A + I_n
        d = np.diag(np.sum(a, 1))  # D^ = Sigma A^_ii
        dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
        at = np.matmul(np.matmul(dinv, a), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
        diff = sp.csr_matrix(args.alpha * inv((np.eye(a.shape[0]) - (1 - args.alpha) * at)))
        print("DIFF TIME!")
        data['diff'] = diff.tocoo()
        data['diff_pos_edges'], data['diff_neg_edges'] = mask_edges_train(diff)
        data['diff_norm'], _ = process(
            data['diff'], data['features'], args.normalize_adj, args.normalize_feats
        )
    
    if args.dataset == 'airport':
        data['features'] = augment(data['adj_train'], data['features'])
        if args.self_supervised != 'non':
            data['features_da'] = augment(data['adj_train'], data['features_da'])
            print("airport3:", data['features'].shape, data['features_da'].shape)
    return data


# ############### FEATURES PROCESSING ####################################


def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features

def process_adj(adj, normalize_adj):
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def process_features(features, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    return features

def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    #return np.array(mask, dtype=np.bool)
    return np.array(mask, dtype=bool)


def fc_prob(args, data):
    num_unknown = data['val_edges'].shape[0] + data['test_edges'].shape[0]
    idx = data['train_edges_false'][:500]
    #idx = data['train_edges'][:500]
    print(idx.shape, idx)
    manifold = getattr(manifolds, args.manifold)()

    x = data['features']
    dim = x.size(-1)
    x_norm = x.view(-1, dim).renorm_(2, 0, 1.)
    
    emb_in = x[idx[:, 0], :]
    emb_out = x[idx[:, 1], :]
    sqdist = manifold.sqdist(emb_in, emb_out, 1.0)
    #sqdist = (emb_in - emb_out).pow(2).sum(dim=-1)
    probs = 1. / (torch.exp((sqdist - 2.0) / 1.0) + 1.0)
    print(probs.shape, probs)
    return probs

def partition(adj, n):
    adj = adj.tocoo()
    node_num = adj.shape[0]
    adj_list = [[] for _ in range(node_num)]
    for i, j in zip(adj.row, adj.col):
        if i == j:
            continue
        adj_list[i].append(j)

    _, ss_labels = metis.part_graph(adjacency=adj_list, nparts=n)
    return np.array(ss_labels)

def completion(features, reduced_dim):
    ss_labels, _, _ = features.svd()   # 奇异���分解，对m*n的矩阵，获得m*m的左�������矩阵
    ss_labels = ss_labels[:, :reduced_dim]
    print(np.argmax(np.array(ss_labels), 1))
    return np.argmax(np.array(ss_labels), 1)

# ############### DATA SPLITS #####################################################

def mask_adjacency(adj, mask_prop, seed=1234):
    np.random.seed(seed)
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)

    n_known = int(len(pos_edges) * (1 - mask_prop))  ### mask掉mask_prop比例的��
    known_edges = pos_edges[:n_known]
    print(n_known)

    masked_adj = sp.csr_matrix((np.ones(known_edges.shape[0]), (known_edges[:, 0], known_edges[:, 1])), shape=adj.shape)
    return masked_adj + masked_adj.T

def complete_adjacency(masked_adj, pseudo_adj, seed=1234):
    plus_adj = masked_adj.toarray() + pseudo_adj

    np.random.seed(seed)
    x, y = sp.triu(plus_adj).nonzero()
    completed_edges = np.array(list(zip(x, y)))
    np.random.shuffle(completed_edges)

    completed_adj = sp.csr_matrix((np.ones(completed_edges.shape[0]), (completed_edges[:, 0], completed_edges[:, 1])), shape=pseudo_adj.shape)
    return completed_adj + completed_adj.T

def pseudo_adjacency(data, seed=1234):
    pseudo_label = data['pseudo_labels'].astype(int) ### no one-hot
    #pseudo_adj = np.zeros((data['features'].shape[0],data['features'].shape[0]), dtype=np.int)
    pseudo_adj = np.zeros((data['features'].shape[0],data['features'].shape[0]), dtype=int)
    
    for i in range(max(pseudo_label)):
        index = np.where(pseudo_label == i)
        for j in index[0]:
            pseudo_adj[np.ix_([j],list(index[0]))] = 1
            pseudo_adj[j][j] = 0
    
    return sp.csr_matrix(pseudo_adj)

def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    print(n_val, n_test)

    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    #train_edges_false = neg_edges[n_test + n_val:]

    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(test_edges_false)  

def mask_edges_pseudo(data, val_prop, test_prop, seed):
    ### adj未���，��造pseudo_adj
    ### 我们认为���标签一致的节点之间边相连（正边���，其余为负边

    pseudo_label = data['pseudo_labels'].astype(int) ### no one-hot
    pseudo_adj = np.zeros((data['features'].shape[0],data['features'].shape[0]), dtype=np.int)
    for i in range(max(pseudo_label)):
        index = np.where(pseudo_label == i)
        for j in index[0]:
            pseudo_adj[np.ix_([j],list(index[0]))] = 1
            pseudo_adj[j][j] = 0

    # get tp edges（true positive）
    np.random.seed(seed)
    x, y = sp.triu(pseudo_adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))   ### 正边（真实的边）
    np.random.shuffle(pos_edges)

    # get tn edges（true negative��������
    x, y = sp.triu(sp.csr_matrix(1. - pseudo_adj)).nonzero()
    neg_edges = np.array(list(zip(x, y)))   ### 负边（不存在的���）
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)   ### ���边总数
    n_val = int(m_pos * val_prop)   ### val���量
    n_test = int(m_pos * test_prop)   ### test数量

    pseudo_val_edges, pseudo_test_edges, pseudo_train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    pseudo_val_edges_false, pseudo_test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    pseudo_train_edges_false = np.concatenate([neg_edges, pseudo_val_edges, pseudo_test_edges], axis=0)  ### 按列进行拼接（构造）
    pseudo_adj_train = sp.csr_matrix((np.ones(pseudo_train_edges.shape[0]), (pseudo_train_edges[:, 0], pseudo_train_edges[:, 1])), shape=pseudo_adj.shape)
    pseudo_adj_train = pseudo_adj_train + pseudo_adj_train.T

    return pseudo_adj_train, torch.LongTensor(pseudo_train_edges), torch.LongTensor(pseudo_train_edges_false), torch.LongTensor(pseudo_val_edges), \
           torch.LongTensor(pseudo_val_edges_false), torch.LongTensor(pseudo_test_edges), torch.LongTensor(pseudo_test_edges_false), pseudo_adj

def mask_edges_train(adj, seed=1234):
    np.random.seed(seed)
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)

    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    return torch.LongTensor(pos_edges), torch.LongTensor(neg_edges)

def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]   ### all labels except label 0
    neg_idx = (1. - labels).nonzero()[0]   ### all labels except label 1
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg

def split_data_new(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    np.random.shuffle(all_idx)
    all_idx = all_idx.tolist()
    nb_val = round(val_prop * len(all_idx))
    nb_test = round(test_prop * len(all_idx))
    idx_val_pos, idx_test_pos, idx_train_pos = all_idx[:nb_val], all_idx[nb_val:nb_val + nb_test], all_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos, idx_test_pos, idx_train_pos


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### LINK PREDICTION DATA LOADERS ####################################


def load_data_lp(dataset, use_feats, data_path):
    if dataset in ['cora', 'pubmed', 'citeseer']:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'amazon':
        adj, features = load_data_amazon(dataset, use_feats, data_path)[:2]
    elif dataset == 'disease':
        dataset = 'disease_lp'
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    elif dataset in ['usa', 'brazil', 'europe']:
        adj, features = load_data_networks(dataset, data_path)[:2]
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = {'adj_train': adj, 'features': features}
    return data


# ############### NODE CLASSIFICATION DATA LOADERS ####################################


def load_data_nc(dataset, use_feats, data_path, split_seed):
    if dataset in ['cora', 'pubmed', 'citeseer']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    elif dataset == 'amazon':
        adj, features, labels, idx_train, idx_val, idx_test = load_data_amazon(
            dataset, use_feats, data_path
        )
    else:
        if dataset == 'disease':
            dataset = 'disease_nc'
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        elif dataset == 'airport':
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        elif dataset in ['usa', 'brazil', 'europe']:
            adj, features, labels = load_data_networks(dataset, data_path)
            val_prop, test_prop = 0.30, 0.65
            #if dataset == 'usa':
            #    val_prop, test_prop = 0.30, 0.65
            #elif dataset == 'brazil':
            #    val_prop, test_prop = 0.15, 0.25
            #elif dataset == 'europe':
            #    val_prop, test_prop = 0.25, 0.55
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)   ### airport: 3364, 524, 524

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data


# ############### DATASETS ####################################


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels


def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph.nodes[u]['feat'] for u in graph.nodes()])   ### (3188, 4)
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features


def load_data_networks(dataset_str, data_path):
    dataset_G = data_path + "/{}.edgelist".format(dataset_str)
    dataset_L = data_path + "/labels-{}.txt".format(dataset_str)
    label_raw, nodes = [], []
    with open(dataset_L, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            node, label = lines.split()
            if label == 'label': continue
            label_raw.append(int(label))
            nodes.append(int(node))
    label_raw = np.array(label_raw)
    print("airport networks label_raw: ", label_raw)
    G = nx.read_edgelist(open(dataset_G, 'rb'), nodetype=int)
    adj = nx.adjacency_matrix(G, nodelist=nodes)

    # task information
    degreeNode = np.sum(adj, axis=1).A1
    degreeNode = degreeNode.astype(np.int32)
    features = np.zeros((degreeNode.size, degreeNode.max() + 1))
    features[np.arange(degreeNode.size), degreeNode] = 1
    print("airport networks features: ", features.shape, features)
    #features = sp.csr_matrix(features)

    return adj, features, label_raw

def load_data_amazon(dataset_str, use_feats, data_path):
    names1 = ['adj_matrix.npz', 'attr_matrix.npz']
    names2 = ['label_matrix.npy', 'train_mask.npy', 'val_mask.npy', 'test_mask.npy']
    objects = []
    for tmp_name in names1:
        tmp_path = os.path.join(data_path, "{}.{}".format(dataset_str, tmp_name))
        objects.append(sp.load_npz(tmp_path))
    for tmp_name in names2:
        tmp_path = os.path.join(data_path, "{}.{}".format(dataset_str, tmp_name))
        objects.append(np.load(tmp_path))
    adj, features, label_matrix, train_mask, val_mask, test_mask = tuple(objects)
    labels = np.argmax(label_matrix, 1)

    arr = np.arange(len(train_mask))
    idx_train = list(arr[train_mask])
    idx_val = list(arr[val_mask])
    idx_test = list(arr[test_mask])

    if not use_feats:
        features = sp.eye(adj.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test
