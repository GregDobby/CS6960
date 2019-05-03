# generate logistic gradients

import pandas as pd
import numpy as np
import argparse
import itertools

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dimension',type=int, help='dimension of feature space')
    parser.add_argument('-b', '--batch',type=int, help='number of points for each worker node')
    parser.add_argument('-S', '--weight_size', type=int, help='number of sampled weigths')
    parser.add_argument('-s', '--size',type=int, help='size of sampled batches for each weight')
    parser.add_argument('-r', '--ratio',type=float, help='ratio of training data')
    parser.add_argument('-w', '--worker',type=int, help='number of workernodes')
    parser.add_argument('-m', '--redundancy',type=int, help='number of redundancy')
    parser.add_argument('-u', '--unreliable',type=int, help='number of unreliable nodes')

    args = parser.parse_args()

    # check arguments
    if (args.weight_size == None or args.batch == None or args.dimension == None or args.size == None or args.ratio == None or args.worker == None or args.redundancy == None or args.unreliable == None):
        parser.print_help()
        return
    dim = args.dimension
    sample_size = args.size
    ratio = args.ratio
    num_worker = args.worker
    redundancy = args.redundancy
    unreliable = args.unreliable
    batch = args.batch
    weight_size = args.weight_size
    
    X = np.empty(shape=[0, num_worker * batch * dim])
    G = np.empty(shape=[0, num_worker * dim])
    W = np.empty(shape=[0, dim])
    for i in range(weight_size):
        w= np.random.multivariate_normal(np.zeros(dim), np.identity(dim))
        x = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), num_worker * batch * sample_size)
        f = 1 / (1+np.exp(-np.matmul(x,w[:,None])))
        g = f * (1 - f) * x
        g = np.reshape(g, [batch, -1])
        g = np.sum(g, axis = 0)
        g = np.reshape(g, [sample_size, -1])
        x = np.reshape(x, [sample_size,-1])
       
        X = np.append(X, x, axis = 0)
        G = np.append(G, g, axis = 0)
        W = np.append(W, np.tile(w, [sample_size, 1]), axis=0)
   
    # np.savetxt('gradient.csv', g, delimiter=',')

    # generate unreliable pattern
    # select unreliable nodes
    mask = np.ones([weight_size * sample_size, num_worker])
    all_comb = np.array(list(itertools.combinations(range(num_worker), unreliable)))
    selected_comb = all_comb[np.random.randint(0,all_comb.shape[0],size=weight_size * sample_size)]
    mask = mask.flatten()
    mask[selected_comb + (num_worker) * np.arange(weight_size * sample_size)[:,None]] = 0
    mask = np.reshape(mask,[weight_size * sample_size, -1])
    mask = np.repeat(mask,dim,axis=1)
    M = G.flatten()[np.where(mask.flatten() == 0)]
    M = np.reshape(M, [-1, dim * unreliable])

    # np.savetxt('mask.csv',mask, delimiter=',')
    # shuffle data first
    index = np.random.permutation(weight_size * sample_size)
    X = X[index,:]
    G = G[index,:]
    W = W[index,:]
    M = M[index,:]
    # split training and testing
    train_size =np.int(np.ceil(weight_size * sample_size * ratio))
    train_x = X[0:train_size, :]
    test_x = X[train_size:,:]
    train_g = G[0:train_size, :]
    test_g = G[train_size:, :]
    train_mask = mask[0:train_size,:]
    test_mask = mask[train_size:,:]
    train_w = W[0:train_size, :]
    test_w = W[train_size:, :]
    
    train_m = M[0:train_size,:]
    test_m = M[train_size:,:]

    np.savetxt('train_x.csv', train_x, delimiter = ',')
    np.savetxt('test_x.csv', test_x, delimiter = ',')
    np.savetxt('train_g.csv', train_g, delimiter = ',')
    np.savetxt('train_mask.csv', train_mask, delimiter =',')
    np.savetxt('test_g.csv', test_g, delimiter=',')
    np.savetxt('test_mask.csv', test_mask, delimiter=',')
    np.savetxt('train_w.csv', train_w, delimiter=',')
    np.savetxt('test_w.csv', test_w, delimiter=',')
    np.savetxt('train_m.csv', train_m, delimiter=',')
    np.savetxt('test_m.csv', test_m, delimiter=',')


if __name__ == '__main__':
    main()
