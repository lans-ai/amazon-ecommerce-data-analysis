# -*- coding: utf-8 -*-
"""
任务2：数据清洗与整合
====================
目标：处理缺失值与异常值，将评分行为数据与商品销售数据通过关键字段（product_id）
整合，生成清洗后的结构化数据集，供后续分析直接使用。

输入（原始数据默认放在工程根目录；可用环境变量 AMAZON_DATA_DIR 覆盖）：
    amazon.csv            —— 商品属性表
    ratings_electronics.csv —— 评分行为表（兼容 Kaggle 原名
                               ratings_Electronics (1).csv）

输出（output/）：
    ratings_clean.csv        —— 评分行为表（7,824,482 行，新增 rating_date）
    amazon_clean.csv         —— 商品主表（1,351 个商品，价格/评分已转数值，含 category_l1、review_count）
    amazon_reviews_long.csv  —— 评论长表（10,598 条，按 user_id/review_id 对齐展开）
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE_DIR, "..", "..")
DATA_DIR = os.environ.get("AMAZON_DATA_DIR", ROOT)


def resolve_data_file(names):
    """在 DATA_DIR 中按候选文件名查找原始数据（规范名优先，兼容 Kaggle 原名）。"""
    for n in names:
        p = os.path.join(DATA_DIR, n)
        if os.path.exists(p):
            return p
    return os.path.join(DATA_DIR, names[0])


RAW_AMAZON = os.path.join(DATA_DIR, "amazon.csv")
RAW_RATINGS = resolve_data_file(
    ["ratings_electronics.csv", "ratings_Electronics (1).csv"]
)
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

RATINGS_COLS = ["user_id", "product_id", "rating", "timestamp"]


def parse_price(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[₹,]", "", regex=True), errors="coerce"
    )


def parse_pct(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"%", "", regex=True), errors="coerce"
    )


def parse_int_comma(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d]", "", regex=True), errors="coerce"
    )


def split_list(s):
    """逗号聚合字段 -> 列表（去空串、去首尾空格）。"""
    if pd.isna(s) or not str(s).strip():
        return []
    return [x.strip() for x in str(s).split(",") if x.strip() != ""]


# ================================================================ 主体
def main():
    print("=" * 70)
    print("任务二：数据清洗与整合")
    print("=" * 70)

    # ---------------- 1. 读入原始数据 ----------------
    amazon = pd.read_csv(RAW_AMAZON, encoding="utf-8", low_memory=False)
    ratings = pd.read_csv(
        RAW_RATINGS,
        header=None,
        names=RATINGS_COLS,
        dtype={
            "user_id": "category",
            "product_id": "category",
            "rating": "float32",
            "timestamp": "int32",
        },
    )
    print(f"[读入] amazon: {len(amazon):,} 行；ratings: {len(ratings):,} 行")

    # ---------------- 2. ratings 清洗 ----------------
    # 缺失/完全重复/评分越界检查（原数据均为 0）
    print(f"[ratings] 缺失值 {ratings.isna().sum().sum():,}；完全重复 {ratings.duplicated().sum():,}")
    oob = int(((ratings["rating"] < 1) | (ratings["rating"] > 5)).sum())
    print(f"[ratings] 评分越界(<1 或 >5) {oob:,} 行（均落在 1-5，无需处理）")

    # 时间戳转日期：Unix 秒 -> rating_date
    ratings["rating_date"] = pd.to_datetime(
        ratings["timestamp"], unit="s", utc=True
    ).dt.date
    print(f"[ratings] 时间范围 {ratings['rating_date'].min()} ~ {ratings['rating_date'].max()}")
    ratings_clean = ratings.copy()
    ratings_clean.to_csv(os.path.join(OUT_DIR, "ratings_clean.csv"),
                         index=False, encoding="utf-8-sig")
    print(f"[ratings] 清洗完成，写入 ratings_clean.csv（{len(ratings_clean):,} 行）")

    # ---------------- 3. amazon 清洗 ----------------
    # 价格/折扣/评分数量：去 ₹、千分位、% 后转数值
    amazon["discounted_price"] = parse_price(amazon["discounted_price"])
    amazon["actual_price"] = parse_price(amazon["actual_price"])
    amazon["discount_percentage"] = parse_pct(amazon["discount_percentage"])
    amazon["rating_count"] = parse_int_comma(amazon["rating_count"])

    # 商品评分异常值：'|' -> NaN（保留行，不整行删除）
    amazon["rating"] = pd.to_numeric(amazon["rating"], errors="coerce")
    bad_rating = int(amazon["rating"].isna().sum())
    print(f"[amazon] rating 异常值 {bad_rating} 行（'|'）置为 NaN")

    # 一级品类：category 全路径取第一段
    amazon["category_l1"] = amazon["category"].str.split("|").str[0].str.strip()

    # 重复商品去重：同一 product_id 保留评分数量最大的行
    before_dedup = len(amazon)
    amazon["_rc_sort"] = amazon["rating_count"].fillna(-1)
    amazon = (amazon.sort_values("_rc_sort", ascending=False)
              .drop_duplicates("product_id", keep="first"))
    amazon = amazon.drop(columns=["_rc_sort"]).reset_index(drop=True)
    print(f"[amazon] 重复商品去重：{before_dedup:,} -> {len(amazon):,} 行")

    # ---------------- 4. 评论长表展开 ----------------
    # 以 user_id/review_id 为锚（两者逐行数量一致），行数严格等于真实评论数；
    # review_title/review_content 因含逗号可能被过度切分，不能作为行数锚点，
    # 否则会产生大量“幽灵行”（缺标题/正文的假象）。
    long_rows = []
    n_name_mismatch = 0
    n_title_mismatch = 0
    n_content_ok = 0
    for _, row in amazon.iterrows():
        uids = split_list(row["user_id"])
        rids = split_list(row["review_id"])
        unames = split_list(row["user_name"])
        rtitles = split_list(row["review_title"])
        contents = split_list(row["review_content"])
        n = len(uids)
        if len(unames) != n:
            n_name_mismatch += 1
        if len(rtitles) != n:
            n_title_mismatch += 1
        if len(contents) == n:
            n_content_ok += 1
        content_ok = len(contents) == n
        for seq in range(1, n + 1):
            uid = uids[seq - 1] if seq - 1 < len(uids) else ""
            rid = rids[seq - 1] if seq - 1 < len(rids) else ""
            uname = unames[seq - 1] if seq - 1 < len(unames) else ""
            rtitle = rtitles[seq - 1] if seq - 1 < len(rtitles) else ""
            content = contents[seq - 1] if (content_ok and seq - 1 < len(contents)) else ""
            long_rows.append(
                {
                    "product_id": row["product_id"],
                    "review_seq": seq,
                    "user_id": uid or "",
                    "user_name": uname or "",
                    "review_id": rid or "",
                    "review_title": rtitle or "",
                    "review_content": content,
                    "content_flag": "ok" if content_ok else "unreliable",
                }
            )
    reviews_long = pd.DataFrame(long_rows)
    print(f"[amazon] 聚合评论字段展开为长表：{len(reviews_long):,} 条评论记录；"
          f"user_name 不一致 {n_name_mismatch} 行、review_title 不一致 {n_title_mismatch} 行；"
          f"review_content 可逐条对齐 {n_content_ok} 行，其余标记 content_flag=unreliable")

    # 派生评论数：按商品统计长表条数，写回商品主表
    review_count = reviews_long.groupby("product_id").size().rename("review_count")
    amazon = amazon.merge(review_count, on="product_id", how="left")
    amazon["review_count"] = amazon["review_count"].fillna(0).astype(int)

    clean_cols = [
        "product_id", "product_name", "category", "category_l1",
        "discounted_price", "actual_price", "discount_percentage",
        "rating", "rating_count", "review_count",
        "about_product", "review_content", "img_link", "product_link",
    ]
    amazon_clean = amazon[clean_cols].copy()
    amazon_clean = amazon_clean.rename(columns={"review_content": "review_content_raw"})
    amazon_clean.to_csv(os.path.join(OUT_DIR, "amazon_clean.csv"),
                        index=False, encoding="utf-8-sig")
    reviews_long.to_csv(os.path.join(OUT_DIR, "amazon_reviews_long.csv"),
                        index=False, encoding="utf-8-sig")
    print(f"[amazon] 清洗完成：商品主表 {len(amazon_clean):,} 行，"
          f"评论长表 {len(reviews_long):,} 行")

    # ---------------- 5. 两表整合 ----------------
    # 两表 product_id 编码体系不同（amazon 为标准 ASIN 形式，ratings 为旧版数字 ID），
    # 直接匹配率极低；ratings 无商品名称字段，无法做名称级模糊匹配。
    # 因此采用两套宽表策略：amazon_clean.csv（商品/销售侧）与 ratings_clean.csv
    # （用户行为侧）各自自洽，匹配上的少量商品仅作跨表验证样本。
    amazon_ids = set(amazon_clean["product_id"].astype(str))
    ratings_ids = set(ratings["product_id"].astype(str))
    matches = sorted(amazon_ids & ratings_ids)
    rate = len(matches) / len(amazon_ids) * 100 if amazon_ids else 0
    print(f"[整合] product_id 精确匹配 {len(matches)} 个（{rate:.2f}%），采用两套宽表策略")
    print("\n完成。")


if __name__ == "__main__":
    main()
