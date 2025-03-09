import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_all_missingness_classification(ds_names):
    n_plots = len(ds_names)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    for ax, ds_name in zip(axes, ds_names):
        df = pd.read_csv(f"{ds_name}_missing_clf_results.csv")
        _plot_missingness_clasification(df, ax)
        ax.set_title(ds_name)
        ax.set_xlabel("Missing Rate")
        ax.set_ylabel("Accuracy")
        ax.set_title(ds_name)
        ax.set_ylim(0.2, 1.05)
        ax.grid(True)
        ax.get_legend().remove()
    fig.suptitle("Accuracy vs. Missing Rate for MAR and MCAR", fontsize=16)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center')
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
    
def plot_fit_pipeline_results(df):
    """
    df: contains the following columns: 'missing_rate', 'seed', 'target', 'omitter', 'imputer', 'regressor', 'mean_score', 'score_std'
    """
    # Get unique values
    missing_rates = sorted(df['missing_rate'].unique())
    targets = sorted(df['target'].unique())
    
    for missing_rate in missing_rates:
        for target in targets:
            # Subset the dataframe for the current missing_rate and target
            sub_df = df[(df['missing_rate'] == missing_rate) & (df['target'] == target)]
            omitters = sorted(sub_df['omitter'].unique())
            regressors = sorted(sub_df['regressor'].unique())
            
            n_rows = len(omitters)
            n_cols = len(regressors)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4), sharex=True)
            
            # Ensure axes is always a 2D array for consistent indexing
            if n_rows == 1 and n_cols == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = np.array([axes])
            elif n_cols == 1:
                axes = np.array([[ax] for ax in axes])
            
            # Loop through each combination of omitter and regressor
            for i, omitter in enumerate(omitters):
                for j, regressor in enumerate(regressors):
                    ax = axes[i, j]
                    # Filter data for the current subplot
                    plot_data = sub_df[(sub_df['omitter'] == omitter) & (sub_df['regressor'] == regressor)]
                    
                    # Optionally, sort by imputer or mean_score for consistency
                    plot_data = plot_data.sort_values(by='imputer')
                    
                    # Create horizontal bar plot
                    ax.barh(
                        plot_data['imputer'],
                        plot_data['mean_score'],
                        xerr=plot_data['score_std'],
                        capsize=4,
                        color='skyblue'
                    )
                    
                    ax.set_title(f"Omitter: {omitter}, Regressor: {regressor}")
                    ax.set_xlabel("Mean Score")
                    ax.set_ylabel("Imputer")
                    
            fig.suptitle(f"Missing Rate: {missing_rate}, Target: {target}", fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('full_exp.csv')
    plot_fit_pipeline_results(df)