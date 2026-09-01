"""
任务1：综合商业分析（第四阶段）
================================
目标：整合前三阶段发现，回答三个关键业务问题：
    ① 哪些品类增长快但差评较多；
    ② 高价值用户的评论关注点（模块化整合）；
    ③ 库存分级建议（热度代理）。

输入（前三阶段产出）：
    product_wide.csv                    —— 商品/销售侧宽表（热度代理：评分数量）
    sentiment_scores.csv                —— 逐条评论情感（VADER/TextBlob）
    topic_sentiment.csv / topic_category.csv / topics.csv —— LDA 主题结果
    user_segment_profile.csv            —— 用户分层画像（高价值用户）
    topn_evaluation.csv                 —— 推荐模型 Top-N 评估

输出：
    output/品类热度差评矩阵.csv / .png —— 业务问题①（增长快但差评多）

口径说明：
    - “增长最快”用「评分数量（rating_count）」做热销/规模代理（amazon 侧无订单日期）；
    - “差评较多”用「VADER 负面占比」（有效评论语料，10,598 条）；
    - 高价值用户（ratings 侧）与评论主题（amazon 侧）无法逐条关联，采用模块化整合。
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

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..", "..")
P1_T3 = os.path.join(ROOT, "01_data_engineering", "task3_metrics", "output")
P2_T1 = os.path.join(ROOT, "02_eda_visualization", "task1_user_sales", "output")
P3_T1 = os.path.join(ROOT, "03_modeling", "task1_sentiment_lda", "output")
P3_T2 = os.path.join(ROOT, "03_modeling", "task2_recommendation", "output")
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  图:", name)


def main():
    print("=" * 70)
    print("任务1：综合商业分析")
    print("=" * 70)

    # ---------------- 1. 品类 × 热度 × 差评矩阵 ----------------
    prod = pd.read_csv(os.path.join(P1_T3, "product_wide.csv"), encoding="utf-8-sig")
    sent = pd.read_csv(
        os.path.join(P3_T1, "02_sentiment", "sentiment_scores.csv"),
        encoding="utf-8-sig",
        usecols=["category_l1", "vader_sent", "tb_sent", "vader_compound"],
    )

    cat_heat = (
        prod.groupby("category_l1")
        .agg(
            商品数=("product_id", "size"),
            评分数量合计=("rating_count", "sum"),
            估算GMV亿=("gmv_est_disc", lambda s: round(s.sum() / 1e8, 1)),
            平均评分=("rating", "mean"),
        )
        .reset_index()
    )
    cat_heat["平均评分"] = cat_heat["平均评分"].round(2)

    cat_sent = (
        sent.groupby("category_l1")
        .agg(
            评论数=("vader_sent", "size"),
            VADER负面占比=("vader_sent", lambda s: (s == "负面").mean() * 100),
            TextBlob负面占比=("tb_sent", lambda s: (s == "负面").mean() * 100),
            平均VADER=("vader_compound", "mean"),
        )
        .reset_index()
    )
    cat_sent["VADER负面占比"] = cat_sent["VADER负面占比"].round(2)
    cat_sent["TextBlob负面占比"] = cat_sent["TextBlob负面占比"].round(2)
    cat_sent["平均VADER"] = cat_sent["平均VADER"].round(3)

    m = cat_heat.merge(cat_sent, on="category_l1", how="left")
    m["评论数"] = m["评论数"].fillna(0).astype(int)
    m["VADER负面占比"] = m["VADER负面占比"].fillna(0)
    m["TextBlob负面占比"] = m["TextBlob负面占比"].fillna(0)
    m["平均VADER"] = m["平均VADER"].fillna(0)

    # 热度等级：评分数量合计 >=100万 高；10万~100万 中；<10万 低
    def heat_tier(x):
        if x >= 1_000_000:
            return "高热度"
        if x >= 10_000:
            return "中热度"
        return "低热度"

    # 差评等级：VADER 负面占比 >=8% 高；6%~8% 中；<6% 低（评论数<100 记“样本小”）
    def neg_tier(row):
        if row["评论数"] < 100:
            return "样本小"
        if row["VADER负面占比"] >= 8:
            return "高差评"
        if row["VADER负面占比"] >= 6:
            return "中差评"
        return "低差评"

    m["热度等级"] = m["评分数量合计"].apply(heat_tier)
    m["差评等级"] = m.apply(neg_tier, axis=1)
    m["定位"] = np.where(
        (m["热度等级"] == "高热度") & (m["差评等级"] == "高差评"), "优先优化",
        np.where(m["热度等级"] == "高热度", "热销关注",
                 np.where(m["差评等级"] == "高差评", "差评预警", "一般观察"))
    )
    m = m.sort_values("评分数量合计", ascending=False).reset_index(drop=True)
    m.to_csv(os.path.join(OUT, "品类热度差评矩阵.csv"), index=False, encoding="utf-8-sig")
    print("[1] 品类热度差评矩阵.csv 完成")

    # 散点图：x=评分数量(log)，y=VADER负面占比，点大小=评论数，颜色=定位
    # 设计要点：小样本（评论<100）用空心标记；标签用贪心避让算法避免压到气泡/其他标签；
    # 坐标轴固定范围让 0 基线与高热度/高差评阈值线有意义；附气泡大小图例。
    color_map = {"优先优化": "#C00000", "热销关注": "#ED7D31", "差评预警": "#7030A0", "一般观察": "#A6A6A6"}
    loc_order = ["优先优化", "热销关注", "差评预警", "一般观察"]
    fig, ax = plt.subplots(figsize=(11.5, 7))

    def bubble_radius(n_reviews):
        return float(np.sqrt(max(n_reviews, 20) * 0.6 / np.pi))

    placed_boxes = []  # (x0, y0, x1, y1) 窗口坐标
    for loc in loc_order:
        sub = m[m["定位"] == loc]
        if sub.empty:
            continue
        small = sub["评论数"] < 100
        ax.scatter(
            np.log10(sub["评分数量合计"].clip(lower=1)),
            sub["VADER负面占比"],
            s=sub["评论数"].clip(lower=20) * 0.6,
            alpha=0.85,
            facecolors=np.where(small, "none", color_map[loc]),
            edgecolors=np.where(small, color_map[loc], "black"),
            linewidths=np.where(small, 1.0, 0.5),
            label=f"{loc}（{len(sub)} 个品类）",
        )

    # 候选标签位置（偏移 points + 水平对齐）
    candidates = [
        (7, 0, "left"), (7, -9, "left"), (7, 9, "left"),
        (-7, 0, "right"), (-7, -9, "right"), (-7, 9, "right"),
        (0, 11, "center"), (0, -11, "center"),
    ]
    anns = {}
    for _, r in m.sort_values("评论数", ascending=False).iterrows():
        px = np.log10(max(r["评分数量合计"], 1))
        py = r["VADER负面占比"]
        br = bubble_radius(r["评论数"])
        best = None
        for dx, dy, ha in candidates:
            ann = ax.annotate(
                r["category_l1"], (px, py),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=9, ha=ha,
            )
            fig.canvas.draw()
            bb = ann.get_window_extent(renderer=fig.canvas.get_renderer())
            # 与已放置标签是否重叠（留 3px 间距）
            if any(bb.expanded(1.08, 1.15).overlaps(pb) for pb in placed_boxes):
                ann.remove()
                continue
            # 与气泡是否重叠（排除自身）
            ok = True
            for r2 in m.itertuples(index=False):
                if r2.category_l1 == r["category_l1"]:
                    continue
                cx, cy = ax.transData.transform(
                    (np.log10(max(r2.评分数量合计, 1)), r2.VADER负面占比)
                )
                label_half = max(bb.width, bb.height) * 0.5
                if np.hypot((bb.x0 + bb.x1) / 2 - cx, (bb.y0 + bb.y1) / 2 - cy) < \
                        bubble_radius(r2.评论数) + label_half * 0.75:
                    ok = False
                    break
            if not ok:
                ann.remove()
                continue
            # 是否在画布内
            if bb.x0 < 20 or bb.x1 > fig.get_size_inches()[0] * fig.dpi - 20 \
                    or bb.y0 < 20 or bb.y1 > fig.get_size_inches()[1] * fig.dpi - 20:
                ann.remove()
                continue
            best = ann
            break
        if best is None:  # 兜底：默认右对齐
            best = ax.annotate(
                r["category_l1"], (px, py),
                xytext=(7, 0), textcoords="offset points", fontsize=9, ha="left",
            )
            fig.canvas.draw()
        bb = best.get_window_extent(renderer=fig.canvas.get_renderer())
        placed_boxes.append(bb)
        anns[r["category_l1"]] = best

    ax.axhline(8, color="#C00000", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(6, color="#ED7D31", linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(np.log10(1_000_000), color="#2E74B5", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(np.log10(1_000_000) + 0.06, 19.2, "高热度线（评分数量 100 万）",
            fontsize=8.5, color="#2E74B5")
    ax.text(3.02, 19.2, "差评阈值：8%（红）/ 6%（橙）", fontsize=8.5, color="#555555")
    ax.set_xlabel("评分数量合计（log10，热度/规模代理）")
    ax.set_ylabel("VADER 负面占比 %")
    ax.set_title("品类×热度×差评矩阵（气泡大小=有效评论数；空心=样本<100 条）")
    ax.set_xlim(3.0, 7.45)
    ax.set_ylim(0, 20)
    ax.set_xticks([3, 4, 5, 6, 7])
    ax.set_xticklabels(["1千", "1万", "10万", "100万", "1000万"])
    ax.set_yticks([0, 5, 10, 15, 20])

    # 气泡大小图例 + 定位图例
    handles = []
    for cnt, lbl in ((100, "评论 100 条"), (1000, "评论 1,000 条"), (3810, "评论 3,810 条")):
        handles.append(ax.scatter([], [], s=cnt * 0.6, facecolors="none",
                                  edgecolors="#888888", linewidths=1, label=lbl))
    leg1 = ax.legend(handles=handles, loc="lower right", title="气泡大小（有效评论数）",
                     fontsize=8.5, title_fontsize=9, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    save_fig(fig, "品类热度差评矩阵.png")

    # ---------------- 2. 主题差评与高价值用户画像（模块化整合） ----------------
    topic_sent = pd.read_csv(
        os.path.join(P3_T1, "03_topics", "topic_sentiment.csv"), encoding="utf-8-sig"
    )
    topics = pd.read_csv(os.path.join(P3_T1, "03_topics", "topics.csv"), encoding="utf-8-sig")
    topic_cat = pd.read_csv(os.path.join(P3_T1, "03_topics", "topic_category.csv"), encoding="utf-8-sig")
    seg = pd.read_csv(os.path.join(P2_T1, "user_segment_profile.csv"), encoding="utf-8-sig")
    topn = pd.read_csv(os.path.join(P3_T2, "02_cf_evaluation", "topn_evaluation.csv"), encoding="utf-8-sig")

    hv = seg[seg["rule_segment"] == "高价值"].iloc[0].to_dict()
    theme_neg = topic_sent.sort_values("负面占比", ascending=False)[
        ["label", "n", "vd_mean", "负面占比"]
    ].copy()
    theme_neg.columns = ["主题", "评论数", "平均VADER", "负面占比"]

    svd20 = topn[(topn["模型"] == "SVD") & (topn["k"] == 20)].iloc[0]
    hyb20 = topn[(topn["模型"] == "SVD+热门先验") & (topn["k"] == 20)].iloc[0]
    pop20 = topn[(topn["模型"] == "热门基线") & (topn["k"] == 20)].iloc[0]

    # ---------------- 3. 库存分级（热度代理） ----------------
    bins = [0, 5_000, 50_000, np.inf]
    labels = ["长尾（<5千）", "腰部（5千~5万）", "热销（≥5万）"]
    prod["档位"] = pd.cut(prod["rating_count"], bins=bins, labels=labels, right=False)
    tier = prod.groupby("档位", observed=True).agg(商品数=("product_id", "size")).reset_index()
    tier_map = {
        "热销（≥5万）": "高备货：保证现货率，按评分数量增速动态补货",
        "腰部（5千~5万）": "正常备货：维持安全库存，结合营销活动弹性补货",
        "长尾（<5千）": "低库存：以销定采，避免积压，采用预售/按需采购",
    }

    # ---------------- 4. 关键结论打印 ----------------
    print("\n[结论①] 品类×热度×差评矩阵（评分数量为热度代理）：")
    for r in m.itertuples(index=False):
        print(f"  {r.category_l1}：评分数量 {r.评分数量合计:,}、VADER 负面占比 {r.VADER负面占比:.1f}%"
              f"（评论 {r.评论数:,} 条）→ {r.热度等级} / {r.差评等级} / {r.定位}")
    print("\n[结论②] 高价值用户画像（营销对象）：")
    print(f"  {hv['用户数']:,} 人（占 {hv['占比%']:.1f}%），人均评分 {hv['n_ratings']:.1f} 次、"
          f"人均商品 {hv['n_products']:.1f} 个、活跃 {hv['active_days']:.0f} 天、平均评分 {hv['avg_rating']:.2f}")
    print("  差评集中主题（产品改进方向）：")
    for r in theme_neg.head(5).itertuples(index=False):
        print(f"    {r.主题}：负面占比 {r.负面占比:.1f}%、平均 VADER {r.平均VADER:.3f}、评论 {r.评论数:,} 条")
    print("\n[结论③] 库存分级建议（热度代理，非精确预测）：")
    for r in tier.itertuples(index=False):
        print(f"  {r.档位}：{r.商品数:,} 个商品 → {tier_map[r.档位]}")
    print("\n[支撑] 推荐模型：SVD MAE 0.792（5 折 CV）；Recall@20 SVD "
          f"{svd20['Recall@k']*100:.1f}% / SVD+热门先验 {hyb20['Recall@k']*100:.1f}% / "
          f"热门基线 {pop20['Recall@k']*100:.1f}%；覆盖率见 topn_evaluation.csv")
    print("\n[完成] 输出已写入 output/")


if __name__ == "__main__":
    main()
