from mkcod_score import mckod_score
from mkc import get_kernel, mkc
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

def MKCOD(X, c, alpha):
    k_tensor = get_kernel(X)
    res = mkc(K=k_tensor, c=c, iter=50, epsilon=1e-4, theta=None)
    out_scores = mckod_score(H=res['H'], alpha=alpha)
    return out_scores

if __name__ == "__main__":
    load_data = loadmat('Example.mat')
    trandata = load_data['Example']
    scaler = MinMaxScaler()
    trandata[:, 0:2] = scaler.fit_transform(trandata[:, 0:2])
    c = 3; alpha=0.5
    anomaly_scores = MKCOD(trandata, c, alpha)
    print(anomaly_scores)