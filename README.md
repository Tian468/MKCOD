# MKCOD

**Jiawei Cui**, Zhong Yuan, Shihao Wang, Yinan Chai, and Ze Zhang, [Multi-kernel clustering-based outlier detection](https://doi.org/10.1016/j.ijar.2026.109800), *International Journal of Approximate Reasoning*, 2026. DOI: [10.1016/j.ijar.2026.109800](https://doi.org/10.1016/j.ijar.2026.109800).

## Abstract

Outlier Detection (OD) has attracted wide attention due to its important applications in financial risk control, cybersecurity, and medical diagnosis. Clustering-based outlier detection identifies outliers by analyzing the relationship between data and clustering structures, offering strong intuitive and interpretable results. However, the performance of such methods heavily depends on clustering results, and they often fail to adequately account for the interference of outliers during clustering, thereby degrading detection performance. Moreover, these methods typically struggle to simultaneously detect different types of outliers, leading to missed outliers. Multi-kernel learning adaptively integrates multiple kernel functions to map data into an appropriate feature space, thereby alleviating the influence of outliers on cluster structures, improving clustering quality, and enhancing outlier detection performance. Based on these observations, this paper proposes a multi-kernel clustering-based outlier detection method. Specifically, a kernel representation composed of multiple kernel functions is first constructed, and the weights of each kernel are adaptively learned during the clustering process to obtain the clustering partition of samples. Then, based on the clustering results, the clusters are divided into large and small clusters. By jointly considering cluster size and sample deviation, an outlier score is defined to enable the simultaneous detection of different types of outlier samples. Experimental comparisons with state-of-the-art methods show that the proposed method achieves competitive AUC performance.

<img width="3328" height="1136" alt="ourwork" src="https://github.com/user-attachments/assets/d4651cdd-1cc6-40bc-8f1e-5c9d49e70fea" />


## Usage

You can run `Demo_MKCOD.py`:

```python
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from mkcod import MKCOD


if __name__ == "__main__":
    X = loadmat("Example.mat")["Example"][:, :2]
    X_train, X_test = train_test_split(
        X, test_size=0.3, random_state=42
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    detector = MKCOD(
        n_clusters=3,
        alpha=0.5,
        rbf_gamma_list=[0.1, 1.0, 10.0]
    )
    detector.fit(X_train)

    anomaly_scores = detector.decision_function(X_test)

    print(anomaly_scores)
```

You can get the anomaly scores as follows:

```text
anomaly_scores =
[1.38711892 0.70256494 1.83110476]
```


## Citation

If you find MKCOD useful in your research, please consider citing:

```bibtex
@article{cui2026multikernel,
  title={Multi-kernel clustering-based outlier detection},
  author={Cui, Jiawei and Yuan, Zhong and Wang, Shihao and Chai, Yinan and Zhang, Ze},
  journal={International Journal of Approximate Reasoning},
  year={2026},
  doi={10.1016/j.ijar.2026.109800}
}
```

## Contact

If you have any questions, please contact [yuanzhong@scu.edu.cn](mailto:yuanzhong@scu.edu.cn).




