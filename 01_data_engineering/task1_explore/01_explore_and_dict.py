# -*- coding: utf-8 -*-
"""
任务1：数据获取与探索
====================
目标：读入两份 Kaggle 原始数据（amazon.csv 商品属性表、ratings_Electronics (1).csv
评分行为表），理解全部字段含义并完成清洗前的质量初检，为任务2 清洗做准备。

输入（原始数据默认放在工程根目录；可用环境变量 AMAZON_DATA_DIR 覆盖）：
    amazon.csv            —— 商品属性表（16 字段）
    ratings_electronics.csv —— 评分行为表（无表头，4 列；兼容 Kaggle 原名
                               ratings_Electronics (1).csv）

输出：
    output/01_data_dictionary.csv —— 数据字典（20 个字段：含义、类型、缺失、唯一值、示例）
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


AMAZON_CSV = os.path.join(DATA_DIR, "amazon.csv")
RATINGS_CSV = resolve_data_file(
    ["ratings_electronics.csv", "ratings_Electronics (1).csv"]
)
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

RATINGS_COLS = ["user_id", "product_id", "rating", "timestamp"]


def load_amazon():
    """读入商品属性表。"""
    return pd.read_csv(AMAZON_CSV, encoding="utf-8", low_memory=False)


def load_ratings():
    """读入评分行为表：无表头；ID 用 category 省内存，评分 float32、时间戳 int32。"""
    return pd.read_csv(
        RATINGS_CSV,
        header=None,
        names=RATINGS_COLS,
        dtype={
            "user_id": "category",
            "product_id": "category",
            "rating": "float32",
            "timestamp": "int32",
        },
    )


def parse_price(series):
    """'₹399' / '₹1,099' -> 399.0 / 1099.0"""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[₹,]", "", regex=True),
        errors="coerce",
    )


def parse_pct(series):
    """'64%' -> 64.0"""
    return pd.to_numeric(
        series.astype(str).str.replace(r"%", "", regex=True),
        errors="coerce",
    )


def parse_int_comma(series):
    """'24,269' -> 24269"""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d]", "", regex=True),
        errors="coerce",
    )


# ================================================================ 探索主体
def main():
    print("=" * 70)
    print("任务一：数据获取与探索")
    print("=" * 70)

    # ---------------- 1. 读入两份数据 ----------------
    amazon = load_amazon()
    ratings = load_ratings()

    print("\n[1] 数据读入完成")
    print(f"  amazon.csv          : {amazon.shape[0]:>10,} 行 × {amazon.shape[1]} 列")
    print(f"  ratings_Electronics : {ratings.shape[0]:>10,} 行 × {ratings.shape[1]} 列")

    # ---------------- 2. amazon.csv 探索 ----------------
    print("\n[2] amazon.csv 概览")
    print("  字段列表：")
    for i, c in enumerate(amazon.columns, 1):
        print(f"    {i:>2}. {c}")
    print(f"  内存占用：{amazon.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")
    print(f"  缺失值合计：{int(amazon.isna().sum().sum())}")
    print(f"  完全重复行：{int(amazon.duplicated().sum())}")

    # 数值字段解析（临时列，只用于质量初检，不写入最终字典）
    amazon["_price_disc"] = parse_price(amazon["discounted_price"])
    amazon["_price_actual"] = parse_price(amazon["actual_price"])
    amazon["_discount_pct"] = parse_pct(amazon["discount_percentage"])
    amazon["_rating_count_num"] = parse_int_comma(amazon["rating_count"])
    amazon["_rating_num"] = pd.to_numeric(amazon["rating"], errors="coerce")
    bad_rating = amazon[amazon["_rating_num"].isna() & amazon["rating"].notna()]
    bad_rating_ids = bad_rating["product_id"].astype(str).tolist()

    print("\n  价格解析结果（discounted_price，失败记 NaN）：")
    print(f"    成功解析 {amazon['_price_disc'].notna().sum():,} / {len(amazon):,}"
          f"，失败 {amazon['_price_disc'].isna().sum():,}")
    print("    折后价：min={:.0f} 中位数={:.0f} max={:,.0f}".format(
        amazon["_price_disc"].min(), amazon["_price_disc"].median(),
        amazon["_price_disc"].max()))

    print("\n  一级品类分布（category 取第一段）：")
    top_cat = amazon["category"].str.split("|").str[0].value_counts()
    for cat, cnt in top_cat.items():
        print(f"    {cat}: {cnt:,}")

    print("\n  商品评分字段 rating 分布：")
    for r, cnt in amazon["rating"].value_counts().sort_index().items():
        print(f"    {r}: {cnt:,}")
    if len(bad_rating) > 0:
        print(f"\n  ⚠ rating 非数值行 {len(bad_rating):,} 行："
              f"product_id = {bad_rating_ids[:5]}（任务二需处理）")

    # 聚合字段检查（逗号分隔的多值单元格）
    multi = amazon["user_id"].astype(str).str.contains(",").sum()
    print(f"\n  聚合字段检查：user_id 含逗号(多值)的行数 {multi:,} / {len(amazon):,}"
          "（评论相关 5 个字段均为逗号聚合，任务二需展开）")

    # ---------------- 3. ratings 探索 ----------------
    print("\n[3] ratings_Electronics (1).csv 概览")
    print(f"  内存占用：{ratings.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")
    print(f"  缺失值：{int(ratings.isna().sum().sum())}")
    print(f"  完全重复行：{int(ratings.duplicated().sum())}")
    print(f"  唯一用户数：{ratings['user_id'].nunique():,}")
    print(f"  唯一商品数：{ratings['product_id'].nunique():,}")
    print("\n  评分分布：")
    for r, cnt in ratings["rating"].value_counts().sort_index().items():
        print(f"    {r}: {cnt:,.0f}（{cnt / len(ratings) * 100:.1f}%）")
    out_of_range = int(((ratings["rating"] < 1) | (ratings["rating"] > 5)).sum())
    print(f"  评分越界(<1 或 >5)行数：{out_of_range:,}")

    ts = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    print(f"  时间范围：{ts.min():%Y-%m-%d} ~ {ts.max():%Y-%m-%d}")
    print(f"  时间戳缺失：{int(ts.isna().sum())}")

    # ---------------- 4. 两表关联现状 ----------------
    overlap = set(amazon["product_id"].astype(str)) & set(ratings["product_id"].astype(str))
    print("\n[4] 两表关联现状（供任务二参考）")
    print(f"  amazon 商品数：{amazon['product_id'].nunique():,}"
          f"；ratings 商品数：{ratings['product_id'].nunique():,}")
    print(f"  product_id 直接匹配数：{len(overlap):,}"
          f"（匹配率 {len(overlap) / amazon['product_id'].nunique() * 100:.2f}%）")

    # ---------------- 5. 数据字典 ----------------
    dict_rows = build_dictionary(amazon, ratings)
    write_dictionary_csv(dict_rows)
    print(f"\n[5] 数据字典已写入 output/01_data_dictionary.csv（{len(dict_rows)} 个字段）")
    print("\n完成。")


# ================================================================ 数据字典
def build_dictionary(amazon, ratings):
    """把人工语义与自动统计合并成数据字典行。"""
    manual = [
        # (数据集, 字段, 业务含义, 类型, 示例, 备注)
        ("ratings", "user_id", "用户唯一标识", "字符串", "A2CX7LUOHB2NDG", "评论行为主体，任务三用户级指标的主键"),
        ("ratings", "product_id", "商品唯一标识（ASIN）", "字符串", "0132793040", "与 amazon.csv 的 product_id 格式不同，直接匹配率极低"),
        ("ratings", "rating", "用户对商品的评分", "浮点(1-5)", "5.0", "5 星占比最高，呈明显右偏"),
        ("ratings", "timestamp", "评分时间（Unix 秒）", "整数", "1365811200", "需转可读日期；约覆盖 1998-12 至 2014-07"),
        ("amazon", "product_id", "商品 ID", "字符串", "B07JW9H4J1", "商品主键，与 ratings 的 ASIN 编码不同"),
        ("amazon", "product_name", "商品名称", "字符串", "Wayona Nylon Braided USB Cable...", "可用于名称规范化匹配"),
        ("amazon", "category", "品类层级", "字符串", "Electronics|...|USBCables", "以 | 分隔；第一段为一级品类"),
        ("amazon", "discounted_price", "折后价", "字符串(₹+千分位)", "₹399", "清洗时去符号转数值"),
        ("amazon", "actual_price", "原价", "字符串(₹+千分位)", "₹1,099", "清洗时去符号转数值"),
        ("amazon", "discount_percentage", "折扣百分比", "字符串", "64%", "清洗时去 % 转数值"),
        ("amazon", "rating", "商品评分（聚合）", "浮点", "4.2", "商品维度评分，非单条用户评分"),
        ("amazon", "rating_count", "评分数量（聚合）", "字符串(千分位)", "24,269", "商品热度/销量代理，清洗时转整数"),
        ("amazon", "about_product", "商品卖点简介", "字符串", "High Compatibility|Fast Charge...", "多条以 | 分隔"),
        ("amazon", "user_id", "用户 ID 列表（聚合）", "字符串", "AG3D6O4STAQK...", "单行内逗号聚合多条，需展开"),
        ("amazon", "user_name", "用户名列表（聚合）", "字符串", "Manav,Adarsh gupta,...", "与 user_id 一一对应"),
        ("amazon", "review_id", "评论 ID 列表（聚合）", "字符串", "R3HXWT0LRP0NMF,...", "单行内逗号聚合多条"),
        ("amazon", "review_title", "评论标题列表（聚合）", "字符串", "Satisfied,Charging is really fast,...", "单行内逗号聚合多条"),
        ("amazon", "review_content", "评论正文列表（聚合）", "字符串", "Looks durable Charging is fine...", "第 2/3 周评论分析素材"),
        ("amazon", "img_link", "商品图片链接", "字符串", "https://m.media-amazon.com/...", "附属信息，一般仅保留"),
        ("amazon", "product_link", "商品详情页链接", "字符串", "https://www.amazon.in/...", "附属信息，一般仅保留"),
    ]

    stat_maps = {}
    for name, df in (("ratings", ratings), ("amazon", amazon)):
        stat_maps[name] = {
            c: {
                "dtype": str(df[c].dtype),
                "missing": int(df[c].isna().sum()),
                "unique": int(df[c].nunique()),
                "sample": str(df[c].dropna().iloc[0])[:42] if df[c].notna().any() else "(空)",
            }
            for c in df.columns
        }

    rows = []
    for ds, field, meaning, ftype, example, note in manual:
        st = stat_maps[ds].get(field, {})
        rows.append(
            {
                "数据集": "ratings_Electronics (1).csv" if ds == "ratings" else "amazon.csv",
                "字段名": field,
                "业务含义": meaning,
                "Pandas类型": st.get("dtype", "-"),
                "缺失数": st.get("missing", "-"),
                "唯一值数": st.get("unique", "-"),
                "示例": example,
                "取值范围/说明": note,
            }
        )
    return rows


def write_dictionary_csv(rows):
    """数据字典写成 CSV（保留字段语义，便于后续分析查阅）。"""
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "01_data_dictionary.csv"),
              index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
