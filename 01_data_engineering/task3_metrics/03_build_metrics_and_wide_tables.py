# -*- coding: utf-8 -*-
"""
任务3：构建分析数据集（业务指标计算 + 宽表生成）
================================================
目标：定义核心业务指标（GMV、复购率、平均评分、热度指数等），并计算生成
可直接用于分析的数据宽表。

输入（任务2 输出）：
    ../task2_clean/output/amazon_clean.csv   —— 商品主表
    ../task2_clean/output/ratings_clean.csv  —— 评分行为表

输出（output/）：
    product_wide.csv  —— 商品销售宽表（1,351 个商品，含估算 GMV、热度指数）
    user_wide.csv     —— 用户行为宽表（420 万用户，含评分次数、活跃跨度、复购标记）
    user_product.csv  —— 用户×商品评分聚合表（782 万对，供第 3 周推荐模型使用）

口径（均为数据可得性下的代理口径）：
    - GMV（估算）= 折后价 × 评分数量（原价口径作对照）
    - 复购率：数据中每个用户×商品仅一次评分，同商品复购恒为 0，采用跨商品复购口径
    - 平均评分：全部评分均值 / 商品级 / 用户级
"""

import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

T0 = time.time()


def tick(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
T2_OUT = os.path.join(BASE_DIR, "..", "task2_clean", "output")
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("任务三：构建分析数据集")
    print("=" * 70)

    # ---------------- 1. 读入清洗后数据 ----------------
    tick("读入 amazon_clean.csv ...")
    amazon = pd.read_csv(
        os.path.join(T2_OUT, "amazon_clean.csv"), encoding="utf-8-sig"
    )
    tick("读入 ratings_clean.csv ...")
    ratings = pd.read_csv(
        os.path.join(T2_OUT, "ratings_clean.csv"),
        encoding="utf-8-sig",
        dtype={"user_id": "category", "product_id": "category",
               "rating": "float32", "timestamp": "int32"},
        usecols=["user_id", "product_id", "rating", "timestamp"],
    )
    print(f"[读入] amazon_clean: {len(amazon):,} 行；ratings_clean: {len(ratings):,} 行")
    ratings["rating_dt"] = pd.to_datetime(ratings["timestamp"], unit="s")

    # ---------------- 2. 用户×商品聚合表 ----------------
    tick("计算 用户×商品 聚合 ...")
    user_product = (
        ratings.groupby(["user_id", "product_id"], observed=True, sort=False)
        .agg(
            n_ratings=("rating", "size"),
            avg_rating=("rating", "mean"),
            first_rating_dt=("rating_dt", "min"),
            last_rating_dt=("rating_dt", "max"),
        )
        .reset_index()
    )
    user_product["is_repeat_same_product"] = user_product["n_ratings"] >= 2
    user_product["first_rating_date"] = user_product["first_rating_dt"].dt.date
    user_product["last_rating_date"] = user_product["last_rating_dt"].dt.date
    user_product = user_product.drop(columns=["first_rating_dt", "last_rating_dt"])
    print(f"[聚合] 用户×商品对数：{len(user_product):,}")

    # ---------------- 3. 商品销售宽表 ----------------
    tick("计算 商品销售宽表 ...")
    # 3.1 指标计算
    amazon["gmv_est_disc"] = (amazon["discounted_price"] * amazon["rating_count"]).round(2)
    amazon["gmv_est_actual"] = (amazon["actual_price"] * amazon["rating_count"]).round(2)
    cat_med = amazon.groupby("category_l1")["rating_count"].transform("median")
    amazon["heat_index"] = (amazon["rating_count"] / cat_med).round(2)

    # 3.2 用 ratings 对商品评分做交叉对照（仅两表匹配商品有值）
    rat_prod = (
        ratings.groupby("product_id", observed=True, sort=False)
        .agg(ratings_avg_rating=("rating", "mean"), ratings_n=("rating", "size"))
        .reset_index()
    )
    product_wide = amazon.merge(rat_prod, on="product_id", how="left")
    product_wide["ratings_avg_rating"] = product_wide["ratings_avg_rating"].round(2)

    product_cols = [
        "product_id", "product_name", "category", "category_l1",
        "discounted_price", "actual_price", "discount_percentage",
        "rating", "rating_count", "review_count",
        "gmv_est_disc", "gmv_est_actual", "heat_index",
        "ratings_avg_rating", "ratings_n",
    ]
    product_wide = product_wide[product_cols]

    # ---------------- 4. 用户行为宽表 ----------------
    tick("计算 用户行为宽表 ...")
    user_agg = (
        ratings.groupby("user_id", observed=True, sort=False)
        .agg(
            n_ratings=("rating", "size"),
            avg_rating=("rating", "mean"),
            rating_std=("rating", "std"),
            first_rating_dt=("rating_dt", "min"),
            last_rating_dt=("rating_dt", "max"),
        )
        .reset_index()
    )
    up_user = (
        user_product.groupby("user_id", observed=True, sort=False)
        .agg(
            n_products=("product_id", "size"),
            n_repeat_same_product=("is_repeat_same_product", "sum"),
        )
        .reset_index()
    )
    user_wide = user_agg.merge(up_user, on="user_id", how="left")
    user_wide["n_products"] = user_wide["n_products"].fillna(0).astype(int)
    user_wide["n_repeat_same_product"] = user_wide["n_repeat_same_product"].fillna(0).astype(int)
    user_wide["is_multi_product"] = user_wide["n_products"] >= 2
    user_wide["is_repeat_same_product"] = user_wide["n_repeat_same_product"] >= 1
    user_wide["active_days"] = (
        user_wide["last_rating_dt"] - user_wide["first_rating_dt"]
    ).dt.days
    user_wide["first_rating_date"] = user_wide["first_rating_dt"].dt.date
    user_wide["last_rating_date"] = user_wide["last_rating_dt"].dt.date
    user_wide = user_wide.drop(columns=["first_rating_dt", "last_rating_dt"])
    user_wide["avg_rating"] = user_wide["avg_rating"].round(2)
    user_wide["rating_std"] = user_wide["rating_std"].round(2)

    # ---------------- 5. 商品级复购率 ----------------
    # 说明：本数据每个用户×商品仅一次评分，该指标恒为 0，仅作口径记录。
    tick("计算 商品级复购率 ...")
    prod_rep = (
        user_product.groupby("product_id", observed=True, sort=False)
        .agg(total_users=("user_id", "size"),
             repeat_users=("is_repeat_same_product", "sum"))
        .reset_index()
    )
    prod_rep["repurchase_rate"] = (
        prod_rep["repeat_users"] / prod_rep["total_users"]
    ).round(4)

    # ---------------- 6. 写文件 ----------------
    tick("写出宽表 CSV ...")
    product_wide.to_csv(os.path.join(OUT_DIR, "product_wide.csv"),
                        index=False, encoding="utf-8-sig")
    user_wide.to_csv(os.path.join(OUT_DIR, "user_wide.csv"),
                     index=False, encoding="utf-8-sig")
    user_product.to_csv(os.path.join(OUT_DIR, "user_product.csv"),
                        index=False, encoding="utf-8-sig")
    print(f"[写出] product_wide: {len(product_wide):,} 行；"
          f"user_wide: {len(user_wide):,} 行；user_product: {len(user_product):,} 行")

    # ---------------- 7. 核心指标摘要（仅打印，供快速核对） ----------------
    n_users = int(ratings["user_id"].nunique())
    n_ratings = len(ratings)
    overall_avg = float(ratings["rating"].mean())
    gmv_disc = float(amazon["gmv_est_disc"].sum())
    repurchase_multi = user_wide["is_multi_product"].mean() * 100
    top_cat = (
        amazon.groupby("category_l1")["gmv_est_disc"].sum()
        .sort_values(ascending=False).head(5)
    )
    print("\n[摘要]")
    print(f"  用户数：{n_users:,}；评分总数：{n_ratings:,}；平均评分：{overall_avg:.2f}")
    print(f"  估算 GMV（折后口径）：₹ {gmv_disc:,.0f}")
    print(f"  跨商品复购率：{repurchase_multi:.2f}%；"
          f"商品级复购率中位数：{prod_rep['repurchase_rate'].median() * 100:.2f}%（数据特性恒为 0）")
    print("  品类估算 GMV 前五（折后口径）：")
    for cat, v in top_cat.items():
        print(f"    {cat}: ₹ {v:,.0f}")
    print("\n完成。")


if __name__ == "__main__":
    main()
