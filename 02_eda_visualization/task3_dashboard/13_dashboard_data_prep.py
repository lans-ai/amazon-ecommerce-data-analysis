# -*- coding: utf-8 -*-
"""
任务3：为 Tableau Public 仪表盘导出聚合/抽样输入表。
====================================================
输入：
    product_wide.csv / user_wide_segmented.csv（01_data_engineering/task3_metrics、
    02_eda_visualization/task1_user_sales 输出）
    ratings_clean.csv / amazon_reviews_long.csv（01_data_engineering/task2_clean 输出）
输出：
    output/dashboard_data/*.csv —— 11 个 Tableau 输入表（KPI、品类、价格带、
                                    热门商品、用户分层、时间趋势、评分分布、评论长度等）
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
P1_T2 = os.path.join(BASE, "..", "..", "01_data_engineering", "task2_clean", "output")
P1_T3 = os.path.join(BASE, "..", "..", "01_data_engineering", "task3_metrics", "output")
T1_OUT = os.path.join(BASE, "..", "task1_user_sales", "output")
OUT = os.path.join(BASE, "output", "dashboard_data")
os.makedirs(OUT, exist_ok=True)


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False, encoding="utf-8-sig")
    print(f"  {name}: {len(df):,} 行")


def main():
    print("=" * 60)
    print("任务3：仪表盘数据准备")
    print("=" * 60)

    prod = pd.read_csv(os.path.join(P1_T3, "product_wide.csv"), encoding="utf-8-sig")
    seg = pd.read_csv(
        os.path.join(T1_OUT, "user_wide_segmented.csv"),
        encoding="utf-8-sig",
        dtype={"user_id": str},
        usecols=["user_id", "n_ratings", "n_products", "active_days",
                 "avg_rating", "rule_segment", "cluster"],
    )
    print(f"[读入] product_wide {len(prod):,}；user_wide_segmented {len(seg):,}")

    kpi = pd.DataFrame({
        "KPI": ["商品数", "用户数", "评分总数", "平均评分", "估算GMV(折后,亿)",
                "跨商品复购率%", "高价值用户占比%"],
        "数值": [
            len(prod), seg["user_id"].nunique(), seg["n_ratings"].sum(),
            round(seg["avg_rating"].mean(), 2),
            round(prod["gmv_est_disc"].sum() / 1e8, 1),
            round((seg["n_products"] >= 2).mean() * 100, 2),
            round((seg["rule_segment"] == "高价值").mean() * 100, 2),
        ],
        "说明": ["商品主表去重后", "ratings 用户数", "评分行为总数", "用户平均评分均值",
                "折后价×评分数量，代理指标", "评分≥2个不同商品", "规则分层（综合得分）"],
    })
    save(kpi, "dash_kpi_overview.csv")

    cat = prod.groupby("category_l1").agg(
        商品数=("product_id", "size"),
        GMV估算亿=("gmv_est_disc", lambda s: round(s.sum() / 1e8, 1)),
        评分数量合计=("rating_count", "sum"),
        平均评分=("rating", "mean"),
    ).reset_index()
    cat["GMV占比%"] = (cat["GMV估算亿"] / cat["GMV估算亿"].sum() * 100).round(2)
    cat["平均评分"] = cat["平均评分"].round(2)
    save(cat, "dash_sales_category.csv")

    bins = [0, 500, 1000, 2000, 5000, 10000, np.inf]
    labels = ["<500", "500-1k", "1k-2k", "2k-5k", "5k-1万", "≥1万"]
    prod["价格带"] = pd.cut(prod["discounted_price"], bins=bins, labels=labels, right=False)
    band = prod.groupby("价格带", observed=True).agg(
        商品数=("product_id", "size"),
        GMV估算亿=("gmv_est_disc", lambda s: round(s.sum() / 1e8, 1)),
    ).reset_index()
    band["价格带"] = band["价格带"].astype(str)
    save(band, "dash_sales_priceband.csv")

    top = prod.nlargest(50, "rating_count")[
        ["product_id", "product_name", "category_l1", "rating", "rating_count", "gmv_est_disc"]
    ].copy()
    top["GMV估算亿"] = (top["gmv_est_disc"] / 1e8).round(2)
    top["商品名称缩写"] = top["product_name"].str[:30]
    save(top[["product_id", "商品名称缩写", "category_l1", "rating", "rating_count", "GMV估算亿"]],
         "dash_top_products.csv")

    seg_sum = seg.groupby("rule_segment").agg(
        用户数=("user_id", "size"),
        人均评分次数=("n_ratings", "mean"),
        人均商品数=("n_products", "mean"),
        平均活跃天数=("active_days", "mean"),
        平均评分=("avg_rating", "mean"),
    ).reset_index()
    seg_sum["占比%"] = (seg_sum["用户数"] / len(seg) * 100).round(2)
    seg_sum[["人均评分次数", "人均商品数", "平均活跃天数", "平均评分"]] = \
        seg_sum[["人均评分次数", "人均商品数", "平均活跃天数", "平均评分"]].round(2)
    save(seg_sum, "dash_users_segment.csv")

    clu_sum = seg.groupby("cluster").agg(
        用户数=("user_id", "size"),
        人均评分次数=("n_ratings", "mean"),
        人均商品数=("n_products", "mean"),
        平均活跃天数=("active_days", "mean"),
        平均评分=("avg_rating", "mean"),
    ).reset_index()
    clu_sum["占比%"] = (clu_sum["用户数"] / len(seg) * 100).round(2)
    clu_sum[["人均评分次数", "人均商品数", "平均活跃天数", "平均评分"]] = \
        clu_sum[["人均评分次数", "人均商品数", "平均活跃天数", "平均评分"]].round(2)
    save(clu_sum, "dash_users_cluster.csv")

    sample = seg.sample(100000, random_state=42)
    save(sample, "dash_user_sample.csv")

    ratings = pd.read_csv(
        os.path.join(P1_T2, "ratings_clean.csv"),
        encoding="utf-8-sig",
        dtype={"timestamp": "int32", "rating": "float32"},
        usecols=["timestamp", "rating"],
    )
    ratings["year"] = pd.to_datetime(ratings["timestamp"], unit="s").dt.year
    trend = ratings.groupby("year").agg(评分次数=("rating", "size"), 平均评分=("rating", "mean")).reset_index()
    trend["平均评分"] = trend["平均评分"].round(2)
    trend = trend[(trend["year"] >= 2000) & (trend["year"] <= 2014)]
    save(trend, "dash_rating_trend.csv")

    star = ratings["rating"].value_counts().sort_index().reset_index()
    star.columns = ["评分", "条数"]
    star["占比%"] = (star["条数"] / star["条数"].sum() * 100).round(2)
    save(star, "dash_rating_star.csv")

    reviews = pd.read_csv(
        os.path.join(P1_T2, "amazon_reviews_long.csv"), encoding="utf-8-sig",
        usecols=["review_title", "review_content", "content_flag"])
    reviews["title_len"] = reviews["review_title"].astype(str).str.len()
    reviews["content_len"] = reviews["review_content"].astype(str).str.len()
    ok = reviews[
        (reviews["content_flag"] == "ok")
        & reviews["review_content"].notna()
        & (reviews["review_content"].astype(str).str.strip().ne(""))
    ]
    length_stat = pd.DataFrame({
        "语料": ["标题(全量)", "正文(可靠子集)"],
        "条数": [len(reviews), len(ok)],
        "平均长度": [round(reviews["title_len"].mean(), 1), round(ok["content_len"].mean(), 1)],
    })
    save(length_stat, "dash_review_length.csv")

    # 情感占比数值来自 12 脚本输出（修正口径后），作为仪表盘静态输入
    rows = []
    for tool, d in (("TextBlob", {"正面": 67.8, "中性": 25.2, "负面": 6.9}),
                    ("VADER", {"正面": 68.5, "中性": 24.0, "负面": 7.5})):
        for s, v in d.items():
            rows.append({"工具": tool, "情感": s, "占比%": v})
    save(pd.DataFrame(rows), "dash_reviews_sentiment.csv")

    print("[完成] 仪表盘输入表已写入 output/dashboard_data/")


if __name__ == "__main__":
    main()
