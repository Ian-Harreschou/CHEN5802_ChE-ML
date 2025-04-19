import matplotlib.pyplot as plt
from chgnet.model import CHGNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from sklearn.inspection import permutation_importance
import shap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
import os
import sys
import csv

SRC_CODE_PATH = os.path.join('~','bin','CHEN5802_ChE-ML','src')

DATA_SET_PATH = os.path.join('..','data','MatPES-PBE-2025.1.json')

DATA_DIR = os.path.join('..','data')

MATERIAL_TYPE = 'all'

sys.path.append(os.path.expanduser(SRC_CODE_PATH))
from data_preprocessing import DataExtracter, Filter, read_json, write_json


class CHGNetEnergyRegressor:
    """
    Pipeline to extract CHGNet embeddings from structures and train a RandomForest
    to predict per‐atom energies.
    """

    def __init__(
        self,
        dataset,
        model_name: str = "0.3.0",
        test_size: float = 0.2,
        random_state: int = 44,
    ):
        """
        Parameters
        ----------
        dataset
            Raw dataset passed directly to DataExtracter.
        model_name
            Which pretrained CHGNet weights to load.
        test_size
            Fraction of data reserved for testing.
        random_state
            Seed for train/test split and RF.
        """
        # extract structures & (per‐atom) energies
        self.data_extractor = DataExtracter(dataset)
        self.structures = self.data_extractor.get_strucs()
        self.energies    = self.data_extractor.get_energies_per_atom()

        # load CHGNet
        self.chgnet = CHGNet.load(model_name=model_name)

        # train/test config
        self.test_size    = test_size
        self.random_state = random_state

        # placeholders
        self.feature_vectors = None
        self.df              = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.best_params = None
        self.rf          = None

    def extract_features(self):
        """Run CHGNet on every structure to get its crystal embedding."""
        self.feature_vectors = []
        if not os.path.exists(os.path.join(DATA_DIR,'features.csv')):
            for idx, struct in enumerate(self.structures):
                try:
                    fea = self.chgnet.predict_structure(
                        struct, return_crystal_feas=True
                    )["crystal_fea"]
                    self.feature_vectors.append(fea)
                    print(f"[{idx+1}/{len(self.structures)}] feature extracted")
                except ValueError as e:
                    print(f"Error extracting features for structure {idx}: {e}")
                    self.feature_vectors.append([np.nan] * self.chgnet.n_features)
                    
            with open(os.path.join(DATA_DIR,'features.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.feature_vectors)
            print(f"Feature vectors saved to {os.path.join(DATA_DIR,'features.csv')}")
        else:
            with open(os.path.join(DATA_DIR,'features.csv'), 'r') as f:
                reader = csv.reader(f)
                self.feature_vectors = list(reader)
            print(f"Feature vectors loaded from {os.path.join(DATA_DIR,'features.csv')}")

    def create_dataframe(self):
        """Build a DataFrame where columns are embedding dims and final col is energy."""
        self.df = pd.DataFrame(self.feature_vectors)
        self.df["energy"] = self.energies
        # dropping NaN rows
        self.df.dropna(inplace=True)

    def split_data(self):
        """Split X/y into train and test sets."""
        X = self.df.drop(columns=["energy"])
        y = self.df["energy"]
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def tune_hyperparameters(
        self,
        param_grid: dict = None,
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 3,
    ):
        """
        Grid‐search over RF params. Stores best_params.
        """
        if param_grid is None:
            param_grid = {
                "n_estimators": [10, 50, 100, 200,250,300,350,400,450,500],
                "max_depth": [None, 10, 20, 30, 50,60,70,80,90,100,110,120],
                "min_samples_split": [2, 5, 10, 20, 30, 40, 50],
                "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }

        base_rf = RandomForestRegressor(random_state=self.random_state)
        gs = GridSearchCV(
            estimator=base_rf,
            param_grid=param_grid,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        gs.fit(self.X_train, self.y_train)
        self.best_params = gs.best_params_
        print("Best parameters:", self.best_params)

    def train(self, params: dict = None):
        """
        Train RF with either `params` or the `best_params` found.
        """
        if params is None:
            if self.best_params is None:
                raise ValueError("No params provided and tune_hyperparameters() not run.")
            params = self.best_params

        self.rf = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=self.random_state,
        )
        self.rf.fit(self.X_train, self.y_train)

    def evaluate(self):
        """
        Compute MAE on train/test and R^2 on test set.
        Returns a dict of metrics.
        """
        if self.rf is None:
            raise ValueError("Model has not been trained yet.")
        y_pred_train = self.rf.predict(self.X_train)
        y_pred_test  = self.rf.predict(self.X_test)

        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae  = mean_absolute_error(self.y_test, y_pred_test)
        test_r2   = r2_score(self.y_test, y_pred_test)

        print(f"Train MAE: {train_mae:.4f} eV/atom")
        print(f"Test  MAE: {test_mae:.4f} eV/atom")
        print(f"Test   R²: {test_r2:.4f}")

        return {
            "train_mae": train_mae,
            "test_mae": test_mae,
            "test_r2": test_r2,
        }

    def run_pipeline(self, param_grid: dict = None, cv: int = 5, n_jobs: int = -1, verbose: int = 4):
        """
        Convenience method: extract → df → split → tune → train → eval.
        Returns evaluation metrics.
        """
        self.extract_features()
        self.create_dataframe()
        self.split_data()
        self.tune_hyperparameters(param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose)
        self.train()
        return self.evaluate()

    def plot_parity(self):
        """Parity plot: Predicted vs. True energies on test set."""
        y_pred = self.rf.predict(self.X_test)
        plt.scatter(self.y_test, y_pred, alpha=0.6)
        mn = min(self.y_test.min(), y_pred.min())
        mx = max(self.y_test.max(), y_pred.max())
        plt.plot([mn, mx], [mn, mx], "--", linewidth=1)
        plt.xlabel("True Energy (eV/atom)")
        plt.ylabel("Predicted Energy (eV/atom)")
        plt.title("Parity Plot")
        plt.savefig(os.path.join(DATA_DIR,'parity_plot.png'))

    def plot_residuals(self):
        """Histogram of residuals on test set."""
        y_pred = self.rf.predict(self.X_test)
        residuals = self.y_test - y_pred
        plt.hist(residuals, bins=30, edgecolor="k", alpha=1)
        plt.xlabel("Residual (eV/atom)")
        plt.ylabel("Frequency")
        plt.title("Residual Distribution")
        plt.savefig(os.path.join(DATA_DIR,'residuals.png'))

    def plot_feature_importance(self, top_n: int = 20):
        """Bar chart of top_n feature importances from RF."""
        importances = self.rf.feature_importances_
        idx = np.argsort(importances)[::-1][:top_n]
        plt.bar(range(top_n), importances[idx], tick_label=idx)
        plt.xlabel("Embedding Dimension Index")
        plt.ylabel("Importance")
        plt.title(f"Top-{top_n} Feature Importances")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR,'feature_importance.png'))

    def plot_permutation_importance(self, top_n: int = 20, n_repeats: int = 10):
        """Bar chart of top_n permutation importances."""
        perm = permutation_importance(
            self.rf, self.X_test, self.y_test,
            n_repeats=n_repeats, random_state=self.random_state
        )
        idx = np.argsort(perm.importances_mean)[::-1][:top_n]
        plt.bar(range(top_n), perm.importances_mean[idx], tick_label=idx)
        plt.xlabel("Embedding Dimension Index")
        plt.ylabel("Permutation Importance")
        plt.title(f"Top-{top_n} Permutation Importances")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR,'permutation_importance.png'))

    def plot_embedding_pca(self):
        """2D PCA of raw CHGNet embeddings, colored by true energy."""
        coords = PCA(n_components=2).fit_transform(self.feature_vectors)
        plt.scatter(coords[:,0], coords[:,1], c=self.energies, s=15)
        plt.colorbar(label="True Energy (eV/atom)")
        plt.title("PCA of CHGNet Embeddings")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.savefig(os.path.join(DATA_DIR,'embedding_pca.png'))

    def plot_embedding_tsne(self, **kwargs):
        """2D t-SNE of embeddings, colored by true energy."""
        coords = TSNE(n_components=2, **kwargs).fit_transform(np.vstack(self.feature_vectors))
        plt.scatter(coords[:,0], coords[:,1], c=self.energies, s=15)
        plt.colorbar(label="True Energy (eV/atom)")
        plt.title("t-SNE of CHGNet Embeddings")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        # plt.show()
        plt.savefig('embedding_tsne.png')

    def plot_cluster_energy_distribution(self, n_clusters: int = 5):
        """KMeans clustering on embeddings + boxplot of energy per cluster."""
        labels = KMeans(n_clusters=n_clusters, random_state=self.random_state).fit_predict(
            np.vstack(self.feature_vectors)
        )
        self.df['cluster'] = labels
        sns.boxplot(x='cluster', y='energy', data=self.df)
        plt.xlabel("Cluster ID")
        plt.ylabel("Energy (eV/atom)")
        plt.title("Energy Distribution per Embedding Cluster")
        plt.savefig('cluster_energy_distribution.png')

    def plot_shap_summary(self, max_display: int = 20):
        """SHAP summary beeswarm for top features."""
        explainer = shap.TreeExplainer(self.rf)
        shap_vals = explainer.shap_values(self.X_test)
        shap.summary_plot(shap_vals, self.X_test, max_display=max_display)
        # save plot
        plt.savefig('shap_summary.png')

if __name__ == "__main__":
    # Example usage
    dataset = read_json(DATA_SET_PATH) # Replace with your dataset path
    filter = Filter(dataset)
    filtered_data = filter.filter(material_type=MATERIAL_TYPE)
    regressor = CHGNetEnergyRegressor(filtered_data)
    metrics = regressor.run_pipeline()
    print(metrics)
    regressor.plot_parity()
    regressor.plot_residuals()
    regressor.plot_feature_importance()
    regressor.plot_permutation_importance()
    regressor.plot_embedding_pca()
    regressor.plot_embedding_tsne()
    regressor.plot_cluster_energy_distribution()
    regressor.plot_shap_summary()