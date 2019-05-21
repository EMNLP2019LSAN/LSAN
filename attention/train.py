import numpy as np
from tqdm import tqdm
def train(attention_model,train_loader,test_loader,criterion,opt,epochs = 5,GPU=True):
    if GPU:
        attention_model.cuda()
    for i in range(epochs):
        print("Running EPOCH",i+1)
        train_loss = []
        prec_k = []
        ndcg_k = []
        for batch_idx, train in enumerate(tqdm(train_loader)):
            opt.zero_grad()
            x, y = train[0].cuda(), train[1].cuda()
            y_pred= attention_model(x)
            loss = criterion(y_pred, y.float())/train_loader.batch_size
            loss.backward()
            opt.step()
            labels_cpu = y.data.cpu().float()
            pred_cpu = y_pred.data.cpu()
            prec = precision_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            prec_k.append(prec)
            ndcg = Ndcg_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            ndcg_k.append(ndcg)
            train_loss.append(float(loss))
        avg_loss = np.mean(train_loss)
        epoch_prec = np.array(prec_k).mean(axis=0)
        epoch_ndcg = np.array(ndcg_k).mean(axis=0)
        print("epoch %2d train end : avg_loss = %.4f" % (i+1, avg_loss))
        print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (epoch_prec[0], epoch_prec[2], epoch_prec[4]))
        print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (epoch_ndcg[0], epoch_ndcg[2], epoch_ndcg[4]))
        test_acc_k = []
        test_loss = []
        test_ndcg_k = []
        for batch_idx, test in enumerate(tqdm(test_loader)):
            x, y = test[0].cuda(), test[1].cuda()
            val_y= attention_model(x)
            loss = criterion(val_y, y.float()) /train_loader.batch_size
            labels_cpu = y.data.cpu().float()
            pred_cpu = val_y.data.cpu()
            prec = precision_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            test_acc_k.append(prec)

            ndcg = Ndcg_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            test_ndcg_k.append(ndcg)
            test_loss.append(float(loss))
        avg_test_loss = np.mean(test_loss)
        test_prec = np.array(test_acc_k).mean(axis=0)
        test_ndcg = np.array(test_ndcg_k).mean(axis=0)
        print("epoch %2d test end : avg_loss = %.4f" % (i+1, avg_test_loss))
        print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (
        test_prec[0], test_prec[2], test_prec[4]))
        print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (test_ndcg[0], test_ndcg[2], test_ndcg[4]))


def precision_k(true_mat, score_mat, k):
    p = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    for k in range(k):
        score_mat = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            score_mat[i][rank_mat[i, :-(k + 1)]] = 0
        score_mat = np.ceil(score_mat)
        #         kk = np.argwhere(score_mat>0)
        mat = np.multiply(score_mat, true_mat)
        #         print("mat",mat)
        num = np.sum(mat, axis=1)
        p[k] = np.mean(num / (k + 1))
    return np.around(p, decimals=4)

def Ndcg_k(true_mat, score_mat, k):
    res = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    label_count = np.sum(true_mat, axis=1)

    for m in range(k):
        y_mat = np.copy(true_mat)
        for i in range(rank_mat.shape[0]):
            y_mat[i][rank_mat[i, :-(m + 1)]] = 0
            for j in range(m + 1):
                y_mat[i][rank_mat[i, -(j + 1)]] /= np.log(j + 1 + 1)

        dcg = np.sum(y_mat, axis=1)
        factor = get_factor(label_count, m + 1)
        ndcg = np.mean(dcg / factor)
        res[m] = ndcg
    return np.around(res, decimals=4)
def get_factor(label_count,k):
    res=[]
    for i in range(len(label_count)):
        n=int(min(label_count[i],k))
        f=0.0
        for j in range(1,n+1):
            f+=1/np.log(j+1)
        res.append(f)
    return np.array(res)