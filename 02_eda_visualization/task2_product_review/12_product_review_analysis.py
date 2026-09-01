"""
任务2：产品与评论分析
====================
输入（第一阶段清洗后数据集）：
    amazon_clean.csv / amazon_reviews_long.csv / ratings_clean.csv
输出：
    output/charts/*.png —— 静态分析图表（评分分布、评论长度、词云、情感分布）
"""

import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# NLTK 数据：优先使用环境变量 NLTK_DATA；Windows 常见安装目录
# （%LOCALAPPDATA%\nltk_data）兜底；缺失资源（停用词、VADER 词典）首次运行自动下载。
import nltk

if os.environ.get("NLTK_DATA"):
    nltk.data.path.insert(0, os.environ["NLTK_DATA"])
_local_nltk = os.path.join(os.environ.get("LOCALAPPDATA", ""), "nltk_data")
if _local_nltk and os.path.isdir(_local_nltk) and _local_nltk not in nltk.data.path:
    nltk.data.path.insert(0, _local_nltk)

for _res, _pkg in (("corpora/stopwords", "stopwords"),
                   ("sentiment/vader_lexicon.zip", "vader_lexicon")):
    try:
        nltk.data.find(_res)
    except LookupError:
        print(f"  [nltk] 下载缺失资源：{_res} ...")
        nltk.download(_pkg)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

from scipy.stats import spearmanr
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from wordcloud import WordCloud

BASE = os.path.dirname(os.path.abspath(__file__))
P1_T2 = os.path.join(BASE, "..", "..", "01_data_engineering", "task2_clean", "output")
OUT = os.path.join(BASE, "output")
CHARTS = os.path.join(OUT, "charts")
os.makedirs(CHARTS, exist_ok=True)

STOP = set(stopwords.words("english"))


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


def clean_tokens(text):
    """去 URL/HTML/非字母，小写，去停用词，过滤单字符。"""
    text = re.sub(r"https?://\S+|www\.\S+", " ", str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    return [w for w in text.split() if w not in STOP and len(w) > 1]


def main():
    print("=" * 60)
    print("任务2：产品与评论分析")
    print("=" * 60)

    # ---------------- 1. 读入 ----------------
    amazon = pd.read_csv(os.path.join(P1_T2, "amazon_clean.csv"), encoding="utf-8-sig")
    reviews = pd.read_csv(os.path.join(P1_T2, "amazon_reviews_long.csv"), encoding="utf-8-sig")
    ratings = pd.read_csv(
        os.path.join(P1_T2, "ratings_clean.csv"),
        encoding="utf-8-sig",
        dtype={"rating": "float32"},
        usecols=["rating"],
    )
    print(f"[读入] amazon {len(amazon):,}；reviews_long {len(reviews):,}；ratings {len(ratings):,}")

    # ---------------- 2. 评分分布 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(amazon["rating"].dropna(), bins=28, color="#2E74B5")
    axes[0].set_title("商品评分分布（商品级）")
    axes[0].set_xlabel("商品评分"); axes[0].set_ylabel("商品数")
    stars = ratings["rating"].value_counts().sort_index()
    axes[1].bar(stars.index.astype(str), stars.values, color="#4472C4")
    axes[1].set_title("用户评分分布（行为级）")
    axes[1].set_xlabel("评分"); axes[1].set_ylabel("评分次数")
    save_fig(fig, "rating_distribution.png")
    star_share = (stars / stars.sum() * 100).round(2)
    print(f"[评分] 商品级均值 {amazon['rating'].mean():.2f}；行为级均值 {ratings['rating'].mean():.2f}")
    print("[评分] 行为级分布：" + "；".join(f"{k} 星 {v:.1f}%" for k, v in star_share.items()))

    # ---------------- 3. 评论长度 ----------------
    reviews["title_len"] = reviews["review_title"].astype(str).str.len()
    ok = reviews[
        (reviews["content_flag"] == "ok")
        & reviews["review_content"].notna()
        & (reviews["review_content"].astype(str).str.strip().ne(""))
    ].copy()
    ok["content_len"] = ok["review_content"].astype(str).str.len()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(reviews["title_len"], bins=40, color="#70AD47")
    axes[0].set_title("评论标题长度分布")
    axes[0].set_xlabel("标题长度"); axes[0].set_ylabel("条数")
    if len(ok):
        axes[1].hist(ok["content_len"], bins=40, color="#ED7D31")
        axes[1].set_title("评论正文长度分布（可靠子集）")
        axes[1].set_xlabel("正文长度"); axes[1].set_ylabel("条数")
    save_fig(fig, "review_length_hist.png")

    # 商品评分档 vs 评论长度
    prod_rating = amazon[["product_id", "rating"]].dropna()
    rl = reviews.merge(prod_rating, on="product_id", how="inner")
    rl["评分档"] = pd.cut(rl["rating"], bins=[0, 3.5, 4.2, 5.1],
                          labels=["低分(<3.5)", "中分(3.5-4.2)", "高分(>4.2)"], right=False)
    band_stat = rl.groupby("评分档", observed=True)["title_len"].agg(["mean", "median", "count"]).round(1)

    band_names = [str(b) for b in band_stat.index]
    x = np.arange(len(band_names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, band_stat["mean"], width, label="均值", color="#2E74B5")
    ax.bar(x + width / 2, band_stat["median"], width, label="中位数", color="#ED7D31")
    ax.set_xticks(x)
    ax.set_xticklabels(band_names)
    ax.set_title("不同商品评分档下的评论标题长度（均值/中位数）")
    ax.set_xlabel("商品评分档"); ax.set_ylabel("标题长度")
    ax.legend()
    save_fig(fig, "review_length_by_rating.png")

    # 商品级：商品评分 vs 平均评论长度 的相关性
    per_prod = rl.groupby("product_id").agg(avg_title_len=("title_len", "mean"), rating=("rating", "first"))
    corr, pval = spearmanr(per_prod["rating"], per_prod["avg_title_len"])
    print(f"[长度] 标题长度均值 {reviews['title_len'].mean():.1f}、中位数 {reviews['title_len'].median():.0f}"
          f"（全量 {fmt_num(len(reviews))} 条）；可靠正文 {fmt_num(len(ok))} 条")
    print(f"[长度] 商品评分与平均评论长度 Spearman 相关系数：{corr:.3f}（p={pval:.3g}）")

    # ---------------- 4. 词云 ----------------
    title_tokens = reviews["review_title"].apply(clean_tokens)
    title_corpus = " ".join(" ".join(t) for t in title_tokens if t)
    wc1 = WordCloud(width=1200, height=600, background_color="white", max_words=120,
                    collocations=False, random_state=42).generate(title_corpus)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc1, interpolation="bilinear"); ax.axis("off")
    ax.set_title(f"评论标题词云（全量 {len(reviews):,} 条）")
    save_fig(fig, "wordcloud_titles.png")

    if len(ok):
        content_tokens = ok["review_content"].apply(clean_tokens)
        content_corpus = " ".join(" ".join(t) for t in content_tokens if t)
        wc2 = WordCloud(width=1200, height=600, background_color="white", max_words=120,
                        collocations=False, random_state=42).generate(content_corpus)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc2, interpolation="bilinear"); ax.axis("off")
        ax.set_title(f"评论正文词云（可靠子集 {len(ok):,} 条）")
        save_fig(fig, "wordcloud_content.png")
    print("[词云] 基于全量标题 + 可靠正文子集生成")

    # ---------------- 5. 情感分析 ----------------
    print("  情感分析中（TextBlob 标题）...")
    reviews["tb_polarity"] = reviews["review_title"].apply(
        lambda t: TextBlob(str(t)).sentiment.polarity)
    print("  情感分析中（VADER 标题）...")
    sia = SentimentIntensityAnalyzer()
    reviews["vader_compound"] = reviews["review_title"].apply(
        lambda t: sia.polarity_scores(str(t))["compound"])

    def bucket(x):
        if x > 0.05:
            return "正面"
        if x < -0.05:
            return "负面"
        return "中性"

    reviews["tb_sent"] = reviews["tb_polarity"].apply(bucket)
    reviews["vader_sent"] = reviews["vader_compound"].apply(bucket)
    tb_share = reviews["tb_sent"].value_counts(normalize=True).reindex(["正面", "中性", "负面"]).fillna(0) * 100
    vd_share = reviews["vader_sent"].value_counts(normalize=True).reindex(["正面", "中性", "负面"]).fillna(0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(reviews["tb_polarity"], bins=30, color="#70AD47")
    axes[0].set_title("TextBlob 情感极性分布（标题）")
    axes[0].set_xlabel("极性（-1~1）"); axes[0].set_ylabel("条数")
    axes[1].hist(reviews["vader_compound"], bins=30, color="#2E74B5")
    axes[1].set_title("VADER 情感得分分布（标题）")
    axes[1].set_xlabel("复合得分（-1~1）"); axes[1].set_ylabel("条数")
    save_fig(fig, "sentiment_distribution.png")

    # 商品评分档 vs 情感
    rl2 = reviews.merge(prod_rating, on="product_id", how="inner")
    rl2["评分档"] = pd.cut(rl2["rating"], bins=[0, 3.5, 4.2, 5.1],
                           labels=["低分(<3.5)", "中分(3.5-4.2)", "高分(>4.2)"], right=False)
    sent_by_band = rl2.groupby("评分档", observed=True)["tb_polarity"].mean().round(3)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([str(b) for b in sent_by_band.index], sent_by_band.values, color="#ED7D31")
    ax.set_title("不同商品评分档下的评论平均情感极性（TextBlob）")
    ax.set_xlabel("商品评分档"); ax.set_ylabel("平均极性")
    save_fig(fig, "sentiment_by_rating_band.png")

    print(f"[情感] TextBlob：正面 {tb_share['正面']:.1f}%、中性 {tb_share['中性']:.1f}%、负面 {tb_share['负面']:.1f}%；"
          f"平均极性 {reviews['tb_polarity'].mean():.3f}")
    print(f"[情感] VADER：正面 {vd_share['正面']:.1f}%、中性 {vd_share['中性']:.1f}%、负面 {vd_share['负面']:.1f}%；"
          f"平均复合得分 {reviews['vader_compound'].mean():.3f}")
    print("[完成] 输出已写入 output/")


if __name__ == "__main__":
    main()
