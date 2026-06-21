from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from mkcod import MKCOD


if __name__ == "__main__":
    X = loadmat("Example.mat")["Example"][:, :2]
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    detector = MKCOD(n_clusters=3, alpha=0.5, rbf_gamma_list=[0.1, 1.0, 10.0])
    detector.fit(X_train)
    anomaly_scores = detector.decision_function(X_test)

    print(anomaly_scores)
