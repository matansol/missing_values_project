
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_missingness_clasification(df):
    df_agg = df[df['mechanism'] == 'MAR'].groupby(["mechanism", "strategy", "missing_rate"])["acc"].agg(["mean", "std"]).reset_index()

    labels = list(df.apply(lambda x: f"{x['mechanism']} ({' '.join(x['strategy'].split('_'))})" 
                        if x['mechanism'] == 'MAR' 
                        else x['mechanism'], axis=1))
    # Plot with seaborn
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df_agg, 
        x="missing_rate", 
        y="acc",
        errorbar='sd',
        hue=labels, 
    )
    plt.axhline(y=0.6, color='black', linestyle='--', label='Our Threshold')

    # Formatting
    plt.xlabel("Missing Rate")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Missing Rate for MAR and MCAR")
    plt.ylim(0, 1)  # Ensure the y-axis stays within [0,1]
    plt.legend(title="Strategy")
    # plt.grid(True)

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