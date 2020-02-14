import collections
import random
import time

import numpy as numpy
import torch
import torch.nn.init as init


def load_data_laychou_lyrics(path):
    with open(path,encoding='utf-8') as f:
        lyrics=f.read()
    lyrics=lyrics.replace('\n',' ').replace('\r',' ')
    lyrics=lyrics[:10000]
    idx_to_char=list(set(lyrics))
    char_to_idx=dict([(char,i) for i ,char in enumerate(idx_to_char)])
    vocab_size=len(char_to_idx)
    lyrics_indices=[char_to_idx[char] for char in lyrics]
    return lyrics_indices,char_to_idx,idx_to_char,vocab_size


"""Function to sample randomly,相邻的两个原始序列不一定相邻"""
def data_iter_random(lyrics_indices,batch_size,num_steps,device=None):
    # 长度为n的序列，最多只能采n-1个char
    num_example=(len(lyrics_indices)-1)//num_steps
    example_indices=[i+num_steps for i in range(num_example)]
    # 随机
    random.shuffle(example_indices)

    def _data(i):
        """
        return char list from i:i+step
        """
        return lyrics_indices[i:i+num_steps]
    if device is None:
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(0,num_example,batch_size):
        batch_indices=example_indices[i:i+batch_size] #取出batchsize个首索引
        X=[_data(j) for j in batch_indices]
        X_=[_data(j) for j in batch_indices]
        yield torch.tensor(X,device=device),torch.tensor(X_,device=device)


"""Function to sample consecutively,相邻的两个原始序列相邻"""
def data_iter_consecutive(lyrics_indices,batch_size,num_steps,device=None):
    if device is None:
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 保留下来的序列长度
    lyrics_len=len(lyrics_indices)//batch_size*batch_size
    lyrics_indices=lyrics_indices[:lyrics_len]
    indices=torch.tensor(lyrics_indices,device=device)
    indices=indices.view(batch_size,-1)
    batch_num=(indices.shape[1]-1)//num_steps #batch的个数
    for i in range(batch_num):
        i=i*num_steps
        X=indices[:,i:i+num_steps]
        X_=indices[:,i+1:i+num_steps+1]
        yield X,X_

def one_hot(x,n_class,dtype=torch.float32):
    """transform a list to one_hot encoding maxtrix"""
    onehot=torch.zeros(x.shape[0],n_class,dtype=dtype,device=x.device)
    onehot.scatter_(1,x.long().view(-1,1),1)  # equal to onehot[1,x[i,0]] = 1
    return onehot

def to_onehot(X,n_class):
    """transform 2D mat(batch_size*num_steps) to num_steps*batch_size*num_class"""
    return [one_hot(X[:,i],n_class) for i in range(X.shape[1])]

# model parameters
def get_rnn_params(num_inputs,num_hiddens,num_outputs,device):
    def _one(shape):
        params=torch.zeros(shape,device=device,dtype=torch.float)
        init.normal_(params,0,0.01)
        #A kind of Tensor that is to be considered a module parameter.
        return torch.nn.Parameter(params)
    # hidden layer
    W_xh=_one((num_inputs,num_hiddens))
    W_hh=_one((num_hiddens,num_hiddens))
    b_h=torch.nn.Parameter(torch.zeros(num_hiddens,device=device))

    #output layer
    W_hq=_one((num_hiddens,num_outputs))
    b_q=torch.nn.Parameter(torch.zeros(num_outputs,device=device))
    return (W_xh,W_hh,b_h,W_hq,b_q)

# create RNN model and forward process
def rnn(inputs,state,params):
    W_xh,W_hh,b_h,W_hq,b_q=params
    H,_=state
    outputs=[]
    for X in inputs:
        H=torch.tanh(torch.mamul(X,W_xh)+torch.matmul(H,W_hh)+b_h)
        Y=torch.matmul(H,W_hq)+b_q
        outputs.append(Y)
    return outputs,(H,)

# initialize original RNN parameters of hidden layer , be careful of shape
def init_rnn_state(batch_size,num_hidden,device):
    return (torch.zeros((batch_size,num_hidden),device=device),)


# gradient cilp, threshold if theta
def grad_clipping(params,thresh,device):
    norm=torch.tensor([0.0],device=device)
    for param in params:
        norm+=(param.grad.data**2).sum()
    norm=norm.sqrt().item()
    if norm > thresh:
        for param in params:
            param.grad.data*=(thresh/norm)
    return

def predict_rnn(prefix,num_chars,rnn,params,init_rnn_state,
                num_hidden,vocab_size,device,idx_to_char,char_to_idx):
    """
    predict next sentence of the Given prefix
    prefix:several chars or words of sentence
    num_chars:len of original sentence
    rnn: net function
    params:
    init_rnn_state: middle hidden layer initialization function
    num_hidden:num of hidden layer 
    vocab_size:size of word_dict
    idx_to_char: get char/word by index, type is list
    char_to_idx: get according index by char, type is dict
    return: predict sentence
    """
    state=init_rnn_state(1,num_hidden,device)
    output=[char_to_idx[prefix[0]]]
    # feed first char
    for c in range(num_chars+len(prefix)-1):
        X=to_onehot(torch.tensor([[output[-1]]],device=device),vocab_size)
        (Y,state)=rnn(X,state,params)
        if t<len(prefix)-1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(Y[0].argmax(dim=1).item())
        
    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device,lyrics_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    # data load model choose
    if is_random_iter:
        data_iter_fn=data_iter_random
    else:
        data_iter_fn=data_iter_consecutive
    params=get_params()
    loss==torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:
            # if sample consecutively, init hidden state after epoch begin
            state=init_rnn_state(batch_size,num_hiddens,device)
        loss_sum,n,start=0.0,0,time.time()
        data_iter=data_iter_fn(lyrics_indices,batch_size,num_steps,device)

        for X,Y in data_iter:
            if is_random_iter: 
                # if sample randomly, init state after minibatch generate
                state=init_rnn_state(batch_size,num_hiddens,device)
            else:
                for s in state:
                    s.detach_() #使用detach函数从计算图分离隐藏状态

            inputs=to_onehot(X,vocab_size)
            (outputs,state)=rnn(input,state,params)
            outputs=torch.cat(outputs,dim=0)#  col cat
            y=torch.flatten(Y.T)
            l=loss(otuputs,y.long())

            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            grad_clipping(params,clipping_theta,device)
            sgd(params,lr,1) # no need to norm
            loss_sum+=l.item()*shape[0]
            n+=y.shape[0]
            
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, device, idx_to_char, char_to_idx))
