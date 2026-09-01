"""
任务1：用户与销售分析
====================
输入：
    product_wide.csv / user_wide.csv（01_data_engineering/task3_metrics 宽表）
    ratings_clean.csv（01_data_engineering/task2_clean 清洗数据，用于时间趋势）
输出：
    output/charts/*.png                 —— 静态分析图表
    output/top_products_*.csv           —— TOP 商品表
    output/user_wide_segmented.csv      —— 用户分层结果（规则 + KMeans 写回）
    output/user_segment_profile.csv     —— 分层画像表
    output/cluster_profile.csv          —— KMeans 聚类画像
    output/rule_vs_cluster_crosstab.csv —— 规则分层 × 聚类交叉表
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

BASE = os.path.dirname(os.path.abspath(__file__))
P1_T3 = os.path.join(BASE, "..", "..", "01_data_engineering", "task3_metrics", "output")
P1_T2 = os.path.join(BASE, "..", "..", "01_data_engineering", "task2_clean", "output")
OUT = os.path.join(BASE, "output")
CHARTS = os.path.join(OUT, "charts")
os.makedirs(CHARTS, exist_ok=True)


def fmt_num(n):
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  图:", name)


def main():
    print("=" * 60)
    print("任务1：用户与销售分析")
    print("=" * 60)

    # ---------------- 1. 读入数据 ----------------
    prod = pd.read_csv(os.path.join(P1_T3, "product_wide.csv"), encoding="utf-8-sig")
    users = pd.read_csv(os.path.join(P1_T3, "user_wide.csv"), encoding="utf-8-sig")
    print(f"[读入] product_wide {len(prod):,}；user_wide {len(users):,}")

    ratings = pd.read_csv(
        os.path.join(P1_T2, "ratings_clean.csv"),
        encoding="utf-8-sig",
        dtype={"timestamp": "int32", "rating": "float32"},
        usecols=["timestamp", "rating"],
    )
    print(f"[读入] ratings {len(ratings):,}（用于时间趋势）")

    # ---------------- 2. 销售侧分析 ----------------
    cat_gmv = (
        prod.groupby("category_l1")
        .agg(
            商品数=("product_id", "size"),
            估算GMV=("gmv_est_disc", "sum"),
            评分数量=("rating_count", "sum"),
            平均评分=("rating", "mean"),
        )
        .sort_values("估算GMV", ascending=False)
    )
    cat_gmv["GMV占比%"] = (cat_gmv["估算GMV"] / cat_gmv["估算GMV"].sum() * 100).round(2)
    cat_gmv = cat_gmv.round(2)

    fig, ax = plt.subplots(figsize=(9, 5))
    top = cat_gmv.head(8)
    ax.bar(top.index, top["估算GMV"] / 1e8, color="#2E74B5")
    ax.set_title("各一级品类估算 GMV（折后口径，亿元）")
    ax.set_xlabel("一级品类")
    ax.set_ylabel("估算 GMV（亿元）")
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(top["估算GMV"] / 1e8):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    save_fig(fig, "category_gmv_bar.png")

    # 价格带分布
    bins = [0, 500, 1000, 2000, 5000, 10000, np.inf]
    labels = ["<500", "500-1k", "1k-2k", "2k-5k", "5k-1万", "≥1万"]
    prod["价格带"] = pd.cut(prod["discounted_price"], bins=bins, labels=labels, right=False)
    band = prod.groupby("价格带", observed=True).agg(
        商品数=("product_id", "size"),
        估算GMV=("gmv_est_disc", "sum"),
    )
    band["GMV占比%"] = (band["估算GMV"] / band["估算GMV"].sum() * 100).round(2)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(band.index.astype(str), band["商品数"], color="#4472C4")
    ax.set_title("商品折后价格带分布（商品数）")
    ax.set_xlabel("价格带（INR）")
    ax.set_ylabel("商品数")
    for i, v in enumerate(band["商品数"]):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)
    save_fig(fig, "price_band_bar.png")

    top_rc = prod.nlargest(20, "rating_count")[
        ["product_id", "product_name", "category_l1", "rating", "rating_count", "gmv_est_disc"]
    ]
    top_gmv = prod.nlargest(20, "gmv_est_disc")[
        ["product_id", "product_name", "category_l1", "rating", "rating_count", "gmv_est_disc"]
    ]
    top_rc.to_csv(os.path.join(OUT, "top_products_by_rating_count.csv"), index=False, encoding="utf-8-sig")
    top_gmv.to_csv(os.path.join(OUT, "top_products_by_gmv.csv"), index=False, encoding="utf-8-sig")

    top3_txt = "；".join(
        f"{i}（{row[2]/1e8:.1f} 亿，{row[5]:.1f}%）"
        for i, row in enumerate(cat_gmv.head(3).itertuples(), 1)
    )
    print(f"[销售侧] 商品数 {fmt_num(len(prod))}；估算 GMV 合计 ₹ {prod['gmv_est_disc'].sum()/1e8:.1f} 亿")
    print(f"[销售侧] 一级品类估算 GMV 前 3：{top3_txt}")

    # ---------------- 3. 用户行为时间趋势 ----------------
    ratings["dt"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings["year"] = ratings["dt"].dt.year
    yearly = ratings.groupby("year").agg(评分次数=("rating", "size"), 平均评分=("rating", "mean"))
    yearly = yearly[(yearly.index >= 2000) & (yearly.index <= 2014)]
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(yearly.index, yearly["评分次数"], color="#70AD47", label="评分次数")
    ax1.set_xlabel("年份"); ax1.set_ylabel("评分次数")
    ax2 = ax1.twinx()
    ax2.plot(yearly.index, yearly["平均评分"], color="#C00000", marker="o", label="平均评分")
    ax2.set_ylabel("平均评分", color="#C00000")
    ax1.set_title("用户评分行为年度趋势（ratings 数据）")
    save_fig(fig, "rating_time_trend.png")
    print(f"[趋势] 评分行为时间范围：{ratings['dt'].min():%Y-%m} ~ {ratings['dt'].max():%Y-%m}")

    # ---------------- 4. 用户基础画像 ----------------
    def band_counts(series, bins, labels):
        """离散档位统计（避免长尾 + 事后 log 轴失真）。"""
        return (
            pd.cut(series, bins=bins, labels=labels, right=False)
            .value_counts()
            .reindex(labels)
            .fillna(0)
            .astype(int)
        )

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    count_bins = [1, 2, 3, 4, 5, 6, 11, 21, np.inf]
    count_labels = ["1", "2", "3", "4", "5", "6-10", "11-20", "21+"]
    for ax, col, color, title in (
        (axes[0, 0], "n_ratings", "#2E74B5", "评分次数分布"),
        (axes[0, 1], "n_products", "#4472C4", "评分商品数分布"),
    ):
        counts = band_counts(users[col], count_bins, count_labels)
        ax.bar(count_labels, counts, color=color)
        ax.set_title(title)
        for i, v in enumerate(counts):
            ax.text(i, v, f"{v/1e4:.0f}万" if v >= 10000 else str(int(v)),
                    ha="center", va="bottom", fontsize=8)

    rating_bins = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.01]
    rating_labels = ["1-1.5", "1.5-2", "2-2.5", "2.5-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5"]
    counts = band_counts(users["avg_rating"], rating_bins, rating_labels)
    axes[1, 0].bar(rating_labels, counts, color="#70AD47")
    axes[1, 0].set_title("平均评分分布")
    for i, v in enumerate(counts):
        axes[1, 0].text(i, v, f"{v/1e4:.0f}万" if v >= 10000 else str(int(v)),
                        ha="center", va="bottom", fontsize=8)

    day_bins = [-0.5, 0.5, 30.5, 90.5, 180.5, 365.5, 730.5, np.inf]
    day_labels = ["0天", "1-30天", "31-90天", "91-180天", "181-365天", "366-730天", ">730天"]
    counts = band_counts(users["active_days"], day_bins, day_labels)
    axes[1, 1].bar(day_labels, counts, color="#ED7D31")
    axes[1, 1].set_title("活跃天数分布")
    axes[1, 1].tick_params(axis="x", rotation=15)
    for i, v in enumerate(counts):
        axes[1, 1].text(i, v, f"{v/1e4:.0f}万" if v >= 10000 else str(int(v)),
                        ha="center", va="bottom", fontsize=8)
    save_fig(fig, "user_metrics_hist.png")
    print(f"[画像] 用户数 {fmt_num(len(users))}；人均评分次数 {users['n_ratings'].mean():.2f}；"
          f"跨商品用户占比 {users['is_multi_product'].mean()*100:.2f}%")

    # ---------------- 5. 用户分层（规则 + KMeans 交叉验证） ----------------
    feats = ["n_ratings", "n_products", "active_days", "avg_rating"]
    # 综合得分：三个活跃维度的百分位排名均值 + 平均评分加成
    for c in ["n_ratings", "n_products", "active_days"]:
        users[c + "_pct"] = users[c].rank(pct=True)
    users["activity_score"] = (
        users["n_ratings_pct"] + users["n_products_pct"] + users["active_days_pct"]
    ) / 3
    users["score_bonus"] = (users["avg_rating"] >= 4.5).astype(float) * 0.15
    users["composite"] = (users["activity_score"] + users["score_bonus"]).round(4)
    q80 = users["composite"].quantile(0.80)
    q50 = users["composite"].quantile(0.50)
    users["rule_segment"] = np.where(
        users["composite"] >= q80, "高价值",
        np.where(users["composite"] >= q50, "中价值", "低价值"))
    users["rule_score"] = users["composite"]

    X = StandardScaler().fit_transform(users[feats].fillna(0))
    best_k, best_s = None, -1
    silhouettes = {}
    sample_idx = np.random.RandomState(42).choice(len(X), size=min(50000, len(X)), replace=False)
    for k in (3, 4, 5):
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        labels = km.fit_predict(X)
        s = silhouette_score(X[sample_idx], labels[sample_idx])
        silhouettes[k] = round(float(s), 4)
        if s > best_s:
            best_s, best_k = s, k
    km_final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    users["cluster"] = km_final.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    plot_df = pd.DataFrame(X2, columns=["PC1", "PC2"])
    plot_df["cluster"] = users["cluster"].values
    plot_df = plot_df.sample(50000, random_state=42)
    fig, ax = plt.subplots(figsize=(9, 6))
    for c in sorted(plot_df["cluster"].unique()):
        sub = plot_df[plot_df["cluster"] == c]
        ax.scatter(sub["PC1"], sub["PC2"], s=3, alpha=0.5, label=f"聚类{c}")
    ax.set_title(f"用户 KMeans 聚类（k={best_k}，PCA 二维投影，抽样 5 万）")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
    save_fig(fig, "kmeans_scatter.png")

    profile = users.groupby("rule_segment")[feats].mean().round(2)
    profile["用户数"] = users.groupby("rule_segment").size()
    profile["占比%"] = (profile["用户数"] / len(users) * 100).round(2)
    profile = profile[["用户数", "占比%", "n_ratings", "n_products", "active_days", "avg_rating"]]
    profile.to_csv(os.path.join(OUT, "user_segment_profile.csv"), encoding="utf-8-sig")

    cluster_profile = users.groupby("cluster")[feats].mean().round(2)
    cluster_profile["用户数"] = users.groupby("cluster").size()
    cluster_profile.to_csv(os.path.join(OUT, "cluster_profile.csv"), encoding="utf-8-sig")

    crosstab = pd.crosstab(users["rule_segment"], users["cluster"])
    crosstab.to_csv(os.path.join(OUT, "rule_vs_cluster_crosstab.csv"), encoding="utf-8-sig")

    users[["user_id", "n_ratings", "n_products", "active_days", "avg_rating",
           "composite", "rule_segment", "cluster"]].to_csv(
        os.path.join(OUT, "user_wide_segmented.csv"), index=False, encoding="utf-8-sig")

    # 高价值 vs 全体对比
    hv = users[users["rule_segment"] == "高价值"]
    compare = pd.DataFrame({
        "指标": ["n_ratings", "n_products", "active_days", "avg_rating"],
        "高价值均值": [hv[c].mean() for c in feats],
        "全体均值": [users[c].mean() for c in feats],
    }).round(2)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(compare))
    ax.bar(x - 0.18, compare["高价值均值"], 0.36, label="高价值用户", color="#ED7D31")
    ax.bar(x + 0.18, compare["全体均值"], 0.36, label="全体用户", color="#A6A6A6")
    ax.set_xticks(x); ax.set_xticklabels(["评分次数", "评分商品数", "活跃天数", "平均评分"])
    ax.set_title("高价值用户与全体用户特征对比")
    ax.legend()
    save_fig(fig, "high_value_vs_all.png")

    print(f"[分层] 综合得分 q80={q80:.3f} / q50={q50:.3f}；KMeans 轮廓系数 {silhouettes}，选 k={best_k}")
    print("[分层] 规则分层画像：")
    print(profile.to_string())
    print(f"[分层] 高价值用户占比 {len(hv)/len(users)*100:.2f}%（{fmt_num(len(hv))} 人）")
    print("[完成] 输出已写入 output/")


if __name__ == "__main__":
    main()
