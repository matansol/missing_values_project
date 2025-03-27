import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_all_missingness_classification(ds_names, save=False):
    n_plots = len(ds_names)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    for ax, ds_name in zip(axes, ds_names):
        df = pd.read_csv(f"experiments/{ds_name}_missing_clf_results.csv")
        _plot_missingness_clasification(df, ax)
        ax.set_title(ds_name)
        ax.set_xlabel("Missing Rate")
        ax.set_ylabel("Accuracy")
        ax.set_title(ds_name)
        ax.set_ylim(0.2, 1.05)
        ax.grid(False)
        ax.get_legend().remove()
    fig.suptitle("Accuracy vs. Missing Rate for MAR and MCAR", fontsize=16)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center')
    if save:
        fig.savefig("images/missing_clf_results.pdf", format='pdf')

def _plot_missingness_clasification(df, ax=None):
    mcar_results = df[df.mechanism == 'MCAR']
    mar_results = df[df.mechanism == 'MAR']
    n_strategies = len(df.strategy.unique())
    # Plot with seaborn
    mar_results['legend_label'] = mar_results['strategy'].apply(lambda x: "MAR w/ "+x.replace('_', ' '))
    sns.lineplot(
        data=mar_results, 
        x="missing_rate", 
        y="acc", 
        hue="legend_label", 
        marker="o",
        linestyle="-",
        ax=ax
    )

    sns.lineplot(
        data=mcar_results, 
        x="missing_rate", 
        y="acc", 
        marker="o",
        linestyle="--",
        color=sns.color_palette()[n_strategies],
        label="MCAR",
        ax=ax
    )

def plot_missingness_clasification(df, ds_name=None):
    plt.figure(figsize=(8, 6))
    
    _plot_missingness_clasification(df)
    
    # Formatting
    plt.xlabel("Missing Rate")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Missing Rate for MAR and MCAR"+(" on "+ds_name if ds_name else ""))
    plt.legend(title="Strategy")
    plt.ylim(0, 1.05)
    plt.grid(True)

    # Show plot
    plt.show()
    
def plot_imputation_scores(df, save=False):
    eps = 1e-5
    strategies = ['none', 'basic', 'double_threshold', 'range_condition', 'nonlinear']
    missing_rates = [0.1, 0.3, 0.5]
    fig, axs = plt.subplots(len(strategies), len(missing_rates), figsize=(20, 10), sharey=True, sharex=True)
    axs = axs.flatten()
    
    min_x, max_x = df.impute_score.min(), df.impute_score.max()

    # Only assign titles for subplots in the top row (first 3 subplots)
    for i, (ax, (strategy, missing_rate)) in enumerate(zip(axs, itertools.product(strategies, missing_rates))):
        if missing_rate == 0.4 and strategy == 'nonlinear':
            print('hi')
        filtered_df = df[(df.strategy == strategy) & (missing_rate - eps <= df.missing_rate) & (df.missing_rate < missing_rate + eps)]
        sns.barplot(
            data=filtered_df,
            x='impute_score',
            y='impute',
            hue='impute',
            orient='h',
            errorbar='ci',
            ax=ax
        )
        if i < len(missing_rates):
            ax.set_title(f'missing rate {missing_rate}')
        else:
            ax.set_title('')
        ax.set_xlim(min_x, max_x)
        if i % len(missing_rates) == 0:
            ax.set_ylabel(strategy.replace('_', ' '))
        plt.tight_layout()
    if save:
        fig.savefig("images/imputations_results.pdf", format='pdf')