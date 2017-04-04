from sklearn import decomposition


class DimReduction:

    # applying PCA dimensionality reduction method on feature vectors
    def pca(self, X):
        pca = decomposition.KernelPCA(n_components=None, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None, alpha=1.0,
                  fit_inverse_transform=False, eigen_solver='auto', tol=0, max_iter=None, remove_zero_eig=False,
                  random_state=None, copy_X=True, n_jobs=1)
        pca.fit(X)
        X = pca.transform(X)
        return X