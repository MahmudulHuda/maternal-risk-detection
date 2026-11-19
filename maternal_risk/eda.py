"""Print the dataset summary used in the README and save the EDA figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from maternal_risk import data


def main():
    cfg = data.load_config()
    figures = data.resolve(cfg["reports_dir"]) / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    raw = data.load_raw(cfg["data_path"])
    print(f"rows: {len(raw)}, missing values: {int(raw.isna().sum().sum())}")
    print(f"exact duplicate rows: {int(raw.duplicated().sum())}")
    print(f"rows with heart rate < {data.MIN_HEART_RATE}: "
          f"{int((raw['HeartRate'] < data.MIN_HEART_RATE).sum())}")
    print("\nclass counts, raw:")
    print(raw[data.TARGET].value_counts().to_string())

    deduplicated = data.clean(raw, drop_conflicts=False)
    labels_per_vector = deduplicated.groupby(data.FEATURES)[data.TARGET].nunique()
    conflicting = labels_per_vector[labels_per_vector > 1]
    print(f"\nrows after removing duplicates and bad heart rates: {len(deduplicated)}")
    print(f"readings that appear with more than one label: {len(conflicting)}")

    df = data.clean(raw)
    print(f"rows after also removing those readings: {len(df)}")
    print("\nclass counts, cleaned:")
    named = df[data.TARGET].map(dict(enumerate(data.LABELS)))
    print(named.value_counts().to_string())
    print("\nfeature summary, cleaned:")
    print(df[data.FEATURES].describe().T.round(2).to_string())

    plot_df = df.assign(risk=named)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, feature in zip(axes.flat, data.FEATURES):
        sns.boxplot(data=plot_df, x="risk", y=feature, order=data.LABELS, ax=ax)
        ax.set_xlabel("")
    fig.suptitle("Features by risk level (cleaned data)")
    fig.tight_layout()
    fig.savefig(figures / "features_by_risk.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df[data.FEATURES].corr(method="spearman"), annot=True, fmt=".2f",
                cmap="vlag", center=0, ax=ax)
    ax.set_title("Spearman correlation between features")
    fig.tight_layout()
    fig.savefig(figures / "feature_correlation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
