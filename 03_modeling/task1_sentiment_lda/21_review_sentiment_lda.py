"""
任务1：评论情感与主题建模（第三阶段）
====================================
目标：在第一阶段清洗后的评论长表基础上，用 TextBlob + VADER 对有效评论做深度
情感分析，并分品类、价格带、商品评分档考察差异；用 Gensim LDA 自动提炼用户
关注主题，输出主题×情感、主题×品类交叉分析。

输入：
    ../01_data_engineering/task2_clean/output/amazon_reviews_long.csv
    ../01_data_engineering/task2_clean/output/amazon_clean.csv

输出（output/ 按类别归档）：
    01_corpus_prep/   —— corpus_prepared.csv
    02_sentiment/     —— sentiment_scores.csv、negative_reviews.csv、情感分布/分维度图表
    03_topics/        —— topics.csv、doc_topics.csv、topic_sentiment.csv、
                         topic_category.csv、主题一致性/占比/关键词/情感图表
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

from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from gensim import corpora, models
from gensim.models import CoherenceModel

BASE = os.path.dirname(os.path.abspath(__file__))
P1_T2 = os.path.join(BASE, "..", "..", "01_data_engineering", "task2_clean", "output")
OUT = os.path.join(BASE, "output")
PREP = os.path.join(OUT, "01_corpus_prep")
SENT = os.path.join(OUT, "02_sentiment")
TOPIC = os.path.join(OUT, "03_topics")
for d in (PREP, SENT, TOPIC):
    os.makedirs(d, exist_ok=True)

# 通用好评词（good/great/nice/best 等）用于情感分析，但会淹没 LDA 主题差异，
# 因此在 LDA 预处理中剔除；功能性停用词与无信息高频词一并删除。
STOP = set(stopwords.words("english")) | {
    "product", "amazon", "item", "one", "two", "get", "got", "would", "also",
    "even", "still", "like", "buy", "bought", "purchase", "purchased", "order",
    "ordered", "review", "really", "much", "many", "im", "ive", "dont", "didnt",
    "cant", "isnt", "wasnt", "thats", "its", "u", "us", "go", "going", "thing",
    "things", "way", "very", "just", "lot", "something", "anything", "nothing",
    "though", "however", "actually", "basically", "sure", "pretty", "little",
    "good", "great", "nice", "best", "ok", "okay", "superb", "awesome", "excellent",
    "perfect", "wonderful", "amazing", "love", "liked", "satisfied", "satisfaction",
    "overall", "gud", "fine", "happy", "better",
}

# 主题标注：具体品类/属性词权重 3，优先决定主题标签；
# 仅当主题没有具体词时才用泛化评价词兜底为「价格与性价比」。
SPECIFIC_TOPIC_KEYWORDS = {
    "充电与线材": ["cable", "charging", "charger", "usb", "wire", "adapter", "fast_charging", "charge", "port"],
    "电池与续航": ["battery", "backup", "drain", "standby"],
    "影音与音质": ["sound", "bass", "earphone", "audio", "headphone", "speaker", "music", "volume"],
    "电视与画质": ["tv", "picture", "lag", "screen", "display", "resolution", "refresh"],
    "手机与影像": ["phone", "camera", "performance", "fingerprint", "gaming", "ram", "storage"],
    "手表与穿戴": ["watch", "band", "strap", "fitness", "heart"],
    "外设与配件": ["mouse", "keyboard", "pen", "remote", "click", "receiver", "dongle"],
    "网络与连接": ["wifi", "router", "internet", "signal", "range", "connection"],
    "家电类": ["heater", "heating", "room", "water", "air", "fan", "purifier", "temperature"],
    "显示器": ["monitor", "panel", "hdmi"],
    "物流与包装": ["delivery", "shipping", "packaging", "received", "delivered", "arrived", "package"],
    "客服与售后": ["service", "support", "warranty", "refund", "return", "replacement"],
    "易用与安装": ["easy", "easy_use", "install", "installation", "setup", "simple", "compatible"],
    "价格与性价比": ["price", "budget", "cost", "cheap", "expensive", "deal", "discount"],
}
EVAL_WORDS = {"quality", "value_money", "worth", "money", "average", "overall"}


def label_topic(words):
    """根据主题 top 词为 LDA 主题打业务标签。"""
    scores = {lab: sum(3 for w in words if w in kws)
              for lab, kws in SPECIFIC_TOPIC_KEYWORDS.items()}
    if not any(scores.values()):
        scores["价格与性价比"] = min(2, sum(1 for w in words if w in EVAL_WORDS))
    if not any(scores.values()):
        scores["综合体验/其他"] = 1
    return max(scores, key=scores.get)


SENT_BUCKETS = ["正面", "中性", "负面"]


def save_fig(fig, name, folder=SENT):
    fig.tight_layout()
    fig.savefig(os.path.join(folder, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  图:", name)


def clean_tokens(text, lemmatizer=None):
    """返回清洗后的 token 列表（去 URL/HTML/非字母、小写、去停用词、可选词形还原）。"""
    if not isinstance(text, str) or not text.strip():
        return []
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    words = [w for w in text.split() if w not in STOP and len(w) > 1]
    if lemmatizer is not None:
        words = [lemmatizer.lemmatize(w) for w in words]
    return words


def bucket(x, pos=0.05, neg=-0.05):
    if x > pos:
        return "正面"
    if x < neg:
        return "负面"
    return "中性"


def sent_share(series):
    return (series.value_counts(normalize=True).reindex(SENT_BUCKETS).fillna(0) * 100).round(2)


def main():
    print("=" * 70)
    print("任务1：评论情感与主题建模")
    print("=" * 70)

    # ---------------- 1. 读入数据 ----------------
    reviews = pd.read_csv(os.path.join(P1_T2, "amazon_reviews_long.csv"), encoding="utf-8-sig")
    amazon = pd.read_csv(os.path.join(P1_T2, "amazon_clean.csv"), encoding="utf-8-sig")
    print(f"[读入] reviews_long {len(reviews):,}；amazon_clean {len(amazon):,}")

    reviews = reviews.merge(
        amazon[["product_id", "category_l1", "discounted_price", "rating"]],
        on="product_id", how="left",
    )

    # ---------------- 2. 语料准备（按 user_id/review_id 对齐口径） ----------------
    reviews["has_title"] = reviews["review_title"].notna() & (
        reviews["review_title"].astype(str).str.strip().ne("")
    )
    reviews["has_content"] = (reviews["content_flag"] == "ok") & reviews["review_content"].notna() & (
        reviews["review_content"].astype(str).str.strip().ne("")
    )

    # 评论文本：优先标题 + 可靠正文合并；仅当至少一个有效文本时保留
    def make_text(r):
        parts = []
        if r["has_title"]:
            parts.append(str(r["review_title"]))
        if r["has_content"]:
            parts.append(str(r["review_content"]))
        return " ".join(parts)

    reviews["text_used"] = reviews.apply(make_text, axis=1)
    reviews["source"] = np.select(
        [reviews["has_title"] & reviews["has_content"], reviews["has_title"], reviews["has_content"]],
        ["标题+正文", "标题", "正文"],
        default="缺失",
    )
    corpus = reviews[reviews["text_used"].str.strip().ne("")].copy()
    print(f"[语料] 有效评论文本：{len(corpus):,} / {len(reviews):,}（缺失 {len(reviews) - len(corpus):,}）")

    # 预处理
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    corpus["tokens"] = corpus["text_used"].apply(lambda t: clean_tokens(t, lemmatizer))
    corpus["token_count"] = corpus["tokens"].apply(len)
    # 情感分析基于全部有效文本；LDA 仅使用分词后有内容的部分（通用好评词过滤后）
    lda_corpus = corpus[corpus["token_count"] > 0]
    print(f"[语料] 有效文本 {len(corpus):,}；LDA 可分词文本 {len(lda_corpus):,}；"
          f"平均词数 {lda_corpus['token_count'].mean():.1f}")

    corpus_out = corpus[
        ["review_id", "product_id", "category_l1", "source", "review_title",
         "review_content", "text_used", "token_count"]
    ].copy()
    corpus_out["tokens_str"] = corpus_out["text_used"].apply(lambda t: " ".join(clean_tokens(t, lemmatizer)))
    corpus_out.to_csv(os.path.join(PREP, "corpus_prepared.csv"), index=False, encoding="utf-8-sig")

    # ---------------- 3. 情感分析（TextBlob + VADER） ----------------
    print("  情感分析中（TextBlob + VADER）...")
    sia = SentimentIntensityAnalyzer()
    corpus["tb_polarity"] = corpus["text_used"].apply(lambda t: TextBlob(str(t)).sentiment.polarity)
    corpus["vader_compound"] = corpus["text_used"].apply(lambda t: sia.polarity_scores(str(t))["compound"])
    corpus["tb_sent"] = corpus["tb_polarity"].apply(bucket)
    corpus["vader_sent"] = corpus["vader_compound"].apply(bucket)

    # 修正口径：仅有效标题 + 可靠正文
    corr_tb = sent_share(corpus["tb_sent"])
    corr_vd = sent_share(corpus["vader_sent"])
    corr_tb_mean = float(corpus["tb_polarity"].mean())
    corr_vd_mean = float(corpus["vader_compound"].mean())
    print(f"[情感] TextBlob {corr_tb.to_dict()}，VADER {corr_vd.to_dict()}")

    sent_cols = [
        "review_id", "product_id", "category_l1", "source", "text_used",
        "tb_polarity", "vader_compound", "tb_sent", "vader_sent",
    ]
    corpus[sent_cols].to_csv(os.path.join(SENT, "sentiment_scores.csv"), index=False, encoding="utf-8-sig")

    # 分维度：品类 / 价格带 / 商品评分档
    price_bins = [0, 500, 1000, 2000, 5000, 10000, np.inf]
    price_labels = ["<500", "500-1k", "1k-2k", "2k-5k", "5k-1万", "≥1万"]
    corpus["价格带"] = pd.cut(corpus["discounted_price"], bins=price_bins, labels=price_labels, right=False)
    corpus["评分档"] = pd.cut(
        corpus["rating"], bins=[0, 3.5, 4.2, 5.1],
        labels=["低分(<3.5)", "中分(3.5-4.2)", "高分(>4.2)"], right=False,
    )
    cat_map = {"Electronics": "Electronics", "Home&Kitchen": "Home&Kitchen",
               "Computers&Accessories": "Computers&Accessories"}
    corpus["品类"] = corpus["category_l1"].map(cat_map).fillna("其他品类")

    def by_group(group_col):
        g = corpus.groupby(group_col, observed=True).agg(
            n=("review_id", "size"),
            tb_mean=("tb_polarity", "mean"),
            vd_mean=("vader_compound", "mean"),
            负面占比=("vader_sent", lambda s: (s == "负面").mean() * 100),
        ).round(3)
        g["n"] = g["n"].astype(int)
        return g.reset_index()

    by_cat = by_group("品类")
    by_price = by_group("价格带")
    by_rating = by_group("评分档")

    # 负面评论清单（两工具至少一个负面，按 VADER 复合得分升序）
    neg = corpus[
        (corpus["tb_sent"] == "负面") | (corpus["vader_sent"] == "负面")
    ].sort_values("vader_compound").copy()
    neg_out = neg[
        ["review_id", "product_id", "category_l1", "review_title", "review_content",
         "tb_polarity", "vader_compound", "tb_sent", "vader_sent"]
    ].copy()
    neg_out.to_csv(os.path.join(SENT, "negative_reviews.csv"), index=False, encoding="utf-8-sig")

    # 图表
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, data, title in (
        (axes[0], corr_tb, "TextBlob 情感分布（修正后）"),
        (axes[1], corr_vd, "VADER 情感分布（修正后）"),
    ):
        ax.bar(SENT_BUCKETS, data.values, color=["#70AD47", "#A6A6A6", "#C00000"])
        ax.set_title(title)
        ax.set_ylabel("占比 %")
        for i, v in enumerate(data.values):
            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    save_fig(fig, "sentiment_share_corrected.png")

    def bar_by_group(df, val_col, title, fname, fmt=".3f"):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(df.iloc[:, 0].astype(str), df[val_col], color="#4472C4")
        ax.set_title(title)
        ax.set_xlabel(df.columns[0]); ax.set_ylabel("平均情感得分")
        for i, v in enumerate(df[val_col]):
            ax.text(i, v, f"{v:{fmt}}", ha="center", va="bottom", fontsize=9)
        save_fig(fig, fname)

    bar_by_group(by_cat, "vd_mean", "各品类平均情感得分（VADER，修正后）", "sentiment_by_category.png")
    bar_by_group(by_price, "vd_mean", "各价格带平均情感得分（VADER，修正后）", "sentiment_by_priceband.png")
    bar_by_group(by_rating, "vd_mean", "各商品评分档平均情感得分（VADER，修正后）", "sentiment_by_ratingband.png")

    # ---------------- 4. LDA 主题建模（按商品聚合文档，缓解短文本稀疏） ----------------
    print("  LDA 主题建模中...")
    prod_tokens = lda_corpus.groupby("product_id")["tokens"].apply(lambda s: [t for d in s for t in d])
    prod_docs = prod_tokens.tolist()
    prod_ids = prod_tokens.index.tolist()
    print(f"[LDA] 商品级文档数 {len(prod_docs):,}（短评论聚合，平均词数 "
          f"{np.mean([len(d) for d in prod_docs]):.1f}）")
    docs = prod_docs
    phrases = models.Phrases(docs, min_count=5, threshold=20)
    bigram = models.phrases.Phraser(phrases)
    docs_bi = [bigram[d] for d in docs]
    dictionary = corpora.Dictionary(docs_bi)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    bow_corpus = [dictionary.doc2bow(d) for d in docs_bi]
    print(f"[LDA] 词典大小 {len(dictionary):,}，文档数 {len(docs_bi):,}")

    k_range = [4, 6, 8, 10, 12, 15]
    coherence_scores = {}

    best_k, best_cv = None, -1
    for k in k_range:
        lda = models.LdaModel(
            bow_corpus, num_topics=k, id2word=dictionary,
            passes=10, iterations=50, random_state=42,
        )
        cm = CoherenceModel(
            model=lda, texts=docs_bi, dictionary=dictionary,
            coherence="c_v", topn=10, processes=1,
        )
        cv = float(cm.get_coherence())
        coherence_scores[k] = round(cv, 4)
        print(f"  K={k}  c_v={cv:.4f}")
        if cv > best_cv:
            best_cv, best_k = cv, k

    print(f"[LDA] 选定 K={best_k}（c_v={best_cv:.4f}）")
    lda_final = models.LdaModel(
        bow_corpus, num_topics=best_k, id2word=dictionary,
        passes=20, iterations=100, random_state=42,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(list(coherence_scores.keys()), list(coherence_scores.values()), marker="o", color="#2E74B5")
    ax.set_xticks(list(coherence_scores.keys()))
    ax.set_xlabel("主题数 K"); ax.set_ylabel("C_v 一致性")
    ax.set_title("LDA 主题一致性（C_v）随 K 变化")
    save_fig(fig, "lda_coherence_curve.png", TOPIC)

    # 主题词表与业务标签
    topic_rows = []
    for tid in range(best_k):
        words = [w for w, p in lda_final.show_topic(tid, topn=12)]
        label = label_topic(words)
        topic_rows.append({"topic_id": tid, "label": label, "top_words": ", ".join(words)})
    topics_df = pd.DataFrame(topic_rows)
    topics_df.to_csv(os.path.join(TOPIC, "topics.csv"), index=False, encoding="utf-8-sig")

    # 商品级文档-主题分布
    prod_topic_rows = []
    for pid, bow in zip(prod_ids, bow_corpus):
        probs = lda_final.get_document_topics(bow, minimum_probability=0.0)
        probs = dict(probs)
        dom = max(probs, key=probs.get)
        prod_topic_rows.append({
            "product_id": pid,
            "dominant_topic": int(dom),
            "dominant_prob": round(probs[dom], 4),
            "topic_dist": ";".join(f"{t}:{round(p, 3)}" for t, p in sorted(probs.items())),
        })
    pt_df = pd.DataFrame(prod_topic_rows)
    pt_df.to_csv(os.path.join(TOPIC, "doc_topics.csv"), index=False, encoding="utf-8-sig")

    # 评论级主题归属 = 所属商品的主题（覆盖全部有效文本）；占比按评论数加权
    rev_topic = corpus[["review_id", "product_id"]].merge(
        pt_df[["product_id", "dominant_topic"]], on="product_id", how="left"
    )
    rev_topic = rev_topic.dropna(subset=["dominant_topic"])
    rev_topic["dominant_topic"] = rev_topic["dominant_topic"].astype(int)
    share = rev_topic["dominant_topic"].value_counts(normalize=True).sort_index() * 100
    topics_df["share"] = topics_df["topic_id"].map(share).round(2)

    # 主题 × 情感
    merged = corpus[["review_id", "product_id", "tb_polarity", "vader_compound", "tb_sent", "vader_sent", "品类"]].merge(
        pt_df[["product_id", "dominant_topic"]], on="product_id", how="left"
    )
    merged = merged.dropna(subset=["dominant_topic"])
    merged["dominant_topic"] = merged["dominant_topic"].astype(int)
    topic_sent = merged.groupby("dominant_topic").agg(
        n=("review_id", "size"),
        tb_mean=("tb_polarity", "mean"),
        vd_mean=("vader_compound", "mean"),
        负面占比=("vader_sent", lambda s: (s == "负面").mean() * 100),
    ).round(3)
    topic_sent["n"] = topic_sent["n"].astype(int)
    topic_sent = topic_sent.reset_index().merge(
        topics_df[["topic_id", "label"]], left_on="dominant_topic", right_on="topic_id", how="left"
    ).drop(columns=["topic_id"])
    topic_sent.to_csv(os.path.join(TOPIC, "topic_sentiment.csv"), index=False, encoding="utf-8-sig")

    # 主题 × 品类
    topic_cat = pd.crosstab(merged["dominant_topic"], merged["品类"], normalize="columns") * 100
    topic_cat = topic_cat.round(1).reset_index()
    topic_cat.to_csv(os.path.join(TOPIC, "topic_category.csv"), index=False, encoding="utf-8-sig")

    # 主题占比图
    tdf = topics_df.sort_values("share", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(tdf) + 2)))
    ax.barh([f"T{int(r.topic_id)} {r.label}" for r in tdf.itertuples()], tdf["share"], color="#2E74B5")
    ax.set_xlabel("文档占比 %"); ax.set_title(f"LDA 主题占比（K={best_k}）")
    for i, v in enumerate(tdf["share"]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
    save_fig(fig, "topic_shares.png", TOPIC)

    # 主题词横向条图（2x3 网格）
    n_rows = int(np.ceil(best_k / 3))
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4.2 * n_rows))
    axes = np.atleast_2d(axes)
    for tid in range(best_k):
        ax = axes.flat[tid]
        words = [(w, p) for w, p in lda_final.show_topic(tid, topn=8)][::-1]
        ax.barh([w for w, p in words], [p for w, p in words], color="#4472C4")
        label = topics_df.loc[tid, "label"]
        ax.set_title(f"T{tid} {label}", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
    for idx in range(best_k, n_rows * 3):
        axes.flat[idx].axis("off")
    fig.suptitle(f"LDA 主题关键词（K={best_k}）", y=1.0, fontsize=13)
    save_fig(fig, "topic_words.png", TOPIC)

    # 主题×情感图
    ts = topic_sent.sort_values("vd_mean")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([f"T{int(r.dominant_topic)} {r.label}" for r in ts.itertuples()], ts["vd_mean"], color="#ED7D31")
    ax.set_ylabel("平均 VADER 复合得分"); ax.set_title("各主题平均情感得分（VADER）")
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, "topic_sentiment.png", TOPIC)

    # 主题×品类堆叠图
    tcp = topic_cat.set_index("dominant_topic")
    fig, ax = plt.subplots(figsize=(10, 5))
    cats = [c for c in tcp.columns if c != "dominant_topic"]
    bottom = np.zeros(len(tcp))
    for c in cats:
        ax.bar([f"T{int(t)}" for t in tcp.index], tcp[c], bottom=bottom, label=c)
        bottom += tcp[c].values
    ax.set_ylabel("占比 %"); ax.set_title("各主题在一级品类中的分布")
    ax.legend()
    save_fig(fig, "topic_category.png", TOPIC)

    # ---------------- 5. 结果摘要（打印关键结论） ----------------
    print("\n[摘要] 语料构成：")
    print(f"  有效文本 {len(corpus):,} 条（有效标题 "
          f"{int(corpus['source'].isin(['标题', '标题+正文']).sum()):,}、可靠正文 "
          f"{int(corpus['source'].isin(['正文', '标题+正文']).sum()):,}）；涉及商品 {corpus['product_id'].nunique():,}")
    print(f"  平均词数（LDA 语料）{lda_corpus['token_count'].mean():.1f}；中位数 {int(lda_corpus['token_count'].median())}")
    print("\n[摘要] 情感（修正口径，有效标题+可靠正文）：")
    print(f"  TextBlob 平均极性 {corr_tb_mean:.3f}；VADER 平均复合得分 {corr_vd_mean:.3f}")
    print(f"  VADER 负面占比 {corr_vd['负面']:.1f}%（低分档 {by_rating.set_index('评分档').loc['低分(<3.5)', '负面占比']:.1f}% vs "
          f"高分档 {by_rating.set_index('评分档').loc['高分(>4.2)', '负面占比']:.1f}%）")
    print("\n[摘要] LDA 主题（K=%d，C_v=%.4f）：" % (best_k, best_cv))
    for r in topics_df.sort_values("share", ascending=False).itertuples(index=False):
        print(f"  T{int(r.topic_id)} {r.label}：占比 {r.share:.1f}%，Top 词 {r.top_words}")
    print("\n[完成] 输出已写入 output/")


if __name__ == "__main__":
    main()
