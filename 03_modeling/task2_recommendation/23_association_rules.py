"""
任务2：构建推荐模型（物品关联规则）
====================================
输入：
    task2_recommendation/output/01_data_prep/ratings_filtered.csv（22 脚本产物）
输出（output/03_association_rules/）：
    association_rules.csv    —— 关联规则（支持度/置信度/提升度）
    看了又看_买了又买.csv     —— 按商品输出的 Top-N 关联推荐
    association_top_rules.png —— 提升度 Top 12 规则图
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
PREP = os.path.join(OUT, "01_data_prep")
RULE = os.path.join(OUT, "03_association_rules")
os.makedirs(RULE, exist_ok=True)

TOP_N_PRODUCTS = 500
MIN_PAIR_COUNT = 50
MIN_CONFIDENCE = 0.05
RELATED_TOP = 5


def main():
    print("=" * 70)
    print("任务2：物品关联规则（买了又买 / 看了又看）")
    print("=" * 70)
    df = pd.read_csv(
        os.path.join(PREP, "ratings_filtered.csv"),
        encoding="utf-8-sig", dtype={"user_id": str, "product_id": str},
        usecols=["user_id", "product_id"],
    )
    print(f"[读入] 过滤后评分 {len(df):,}；用户 {df['user_id'].nunique():,}；商品 {df['product_id'].nunique():,}")

    # 取热门商品作为候选项集（篮子分析控制在可计算规模）
    prod_count = df.groupby("product_id", sort=False)["user_id"].size()
    top = prod_count.nlargest(TOP_N_PRODUCTS).index
    baskets = df[df["product_id"].isin(top)][["user_id", "product_id"]].drop_duplicates()
    n_users = baskets["user_id"].nunique()
    print(f"[篮子] Top{TOP_N_PRODUCTS} 商品内的用户 {n_users:,}")

    # 自连接计算两两共现
    m = baskets.merge(baskets, on="user_id")
    m = m[m["product_id_x"] < m["product_id_y"]]
    pairs = m.groupby(["product_id_x", "product_id_y"], sort=False).size().rename("count").reset_index()
    pairs = pairs[pairs["count"] >= MIN_PAIR_COUNT]
    print(f"[共现] 满足最小支持度的商品对：{len(pairs):,}")

    pc = prod_count.rename("count_a")
    pairs = pairs.merge(pc, left_on="product_id_x", right_index=True, how="left")
    pc2 = prod_count.rename("count_b")
    pairs = pairs.merge(pc2, left_on="product_id_y", right_index=True, how="left")
    pairs["support"] = (pairs["count"] / n_users).round(6)
    pairs["confidence_a_b"] = (pairs["count"] / pairs["count_a"]).round(4)
    pairs["support_b"] = pairs["count_b"] / n_users
    pairs["lift"] = (pairs["confidence_a_b"] / pairs["support_b"]).round(3)
    rules = pairs[pairs["confidence_a_b"] >= MIN_CONFIDENCE].sort_values("lift", ascending=False)
    print(f"[规则] 置信度≥{MIN_CONFIDENCE} 的规则：{len(rules):,}")
    rules.to_csv(os.path.join(RULE, "association_rules.csv"), index=False, encoding="utf-8-sig")

    # 按商品输出 Top-N 关联推荐（双向）
    related = {}
    for row in rules.itertuples(index=False):
        related.setdefault(row.product_id_x, []).append((row.product_id_y, row.lift, row.confidence_a_b))
        related.setdefault(row.product_id_y, []).append((row.product_id_x, row.lift, row.confidence_a_b))
    rows = []
    for pid in top[:200]:
        items = sorted(related.get(pid, []), key=lambda x: (-x[1], -x[2]))[:RELATED_TOP]
        for rank, (other, lift, conf) in enumerate(items, 1):
            rows.append({"商品": pid, "关联位次": rank, "关联商品": other,
                         "提升度": lift, "置信度": conf})
    rel_df = pd.DataFrame(rows)
    rel_df.to_csv(os.path.join(RULE, "看了又看_买了又买.csv"), index=False, encoding="utf-8-sig")
    print(f"[输出] 看了又看_买了又买.csv（{len(rel_df):,} 行）")

    # 图表：提升度 Top 12 规则
    top_rules = rules.head(12)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [f"{r.product_id_x} → {r.product_id_y}" for r in top_rules.itertuples(index=False)][::-1]
    vals = top_rules["lift"].tolist()[::-1]
    ax.barh(labels, vals, color="#2E74B5")
    ax.set_xlabel("提升度 Lift")
    ax.set_title("关联规则 Top 12（按提升度）")
    for i, v in enumerate(vals):
        ax.text(v + 0.05, i, f"{v:.1f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RULE, "association_top_rules.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  图: association_top_rules.png")

    # 摘要打印
    print("\n[摘要] 提升度 Top 10 规则：")
    for r in rules.head(10).itertuples(index=False):
        print(f"  {r.product_id_x} -> {r.product_id_y}：共现 {r.count:,}、支持度 {r.support:.5f}、"
              f"置信度 {r.confidence_a_b:.3f}、提升度 {r.lift:.2f}")
    print("[完成] 输出已写入 output/")


if __name__ == "__main__":
    main()
