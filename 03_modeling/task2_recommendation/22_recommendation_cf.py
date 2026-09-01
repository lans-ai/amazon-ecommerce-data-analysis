"""
任务2：构建推荐模型（协同过滤部分）
====================================
输入：
    ../01_data_engineering/task3_metrics/output/user_product.csv
输出（output/ 按类别归档）：
    01_data_prep/     —— ratings_filtered.csv、train.csv、test.csv
    02_cf_evaluation/ —— model_comparison.csv、topn_evaluation.csv、
                         recommendations_sample.csv、评估图表
    03_association_rules/ —— 由 23 脚本产出（association_rules.csv 等）
    04_cold_start/    —— popularity_baseline.csv、new_item_examples.csv
"""

import os
import sys
import time

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

from surprise import BaselineOnly, Dataset, KNNBasic, Reader, SVD
from surprise.model_selection import cross_validate

BASE = os.path.dirname(os.path.abspath(__file__))
P1_T3 = os.path.join(BASE, "..", "..", "01_data_engineering", "task3_metrics", "output")
OUT = os.path.join(BASE, "output")
PREP = os.path.join(OUT, "01_data_prep")
EVAL = os.path.join(OUT, "02_cf_evaluation")
COLD = os.path.join(OUT, "04_cold_start")
for d in (PREP, EVAL, COLD):
    os.makedirs(d, exist_ok=True)

MIN_USER_RATINGS = 5
MIN_PROD_RATINGS = 10
TOP_PRODUCTS_FOR_CV = 3000
CV_SAMPLE_SIZE = 200_000
USER_KNN_SAMPLE = 5000
TOP_N_TEST_USERS = 5000
EVAL_CHUNK = 1000
KS = (5, 10, 20)


def save_fig(fig, name, folder=EVAL):
    fig.tight_layout()
    fig.savefig(os.path.join(folder, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  图:", name)


def load_data():
    print("[1] 读入 user_product.csv ...")
    up = pd.read_csv(
        os.path.join(P1_T3, "user_product.csv"),
        encoding="utf-8-sig",
        dtype={"user_id": str, "product_id": str},
        usecols=["user_id", "product_id", "avg_rating", "last_rating_date"],
    )
    up = up.rename(columns={"avg_rating": "rating"})
    up["rating"] = up["rating"].astype("float32")
    up["last_rating_date"] = pd.to_datetime(up["last_rating_date"])
    print(f"[读入] {len(up):,} 条评分；用户 {up['user_id'].nunique():,}；商品 {up['product_id'].nunique():,}")
    return up


def filter_subset(up):
    print("[2] 过滤子集（用户≥5 条评分，商品≥10 人评分）...")
    u_cnt = up.groupby("user_id", sort=False)["product_id"].size()
    df = up[up["user_id"].isin(u_cnt[u_cnt >= MIN_USER_RATINGS].index)]
    p_cnt = df.groupby("product_id", sort=False)["user_id"].size()
    df = df[df["product_id"].isin(p_cnt[p_cnt >= MIN_PROD_RATINGS].index)]
    u_cnt2 = df.groupby("user_id", sort=False)["product_id"].size()
    df = df[df["user_id"].isin(u_cnt2[u_cnt2 >= MIN_USER_RATINGS].index)]
    df = df.reset_index(drop=True)
    df.to_csv(os.path.join(PREP, "ratings_filtered.csv"), index=False, encoding="utf-8-sig")
    print(f"[过滤后] {len(df):,} 条评分；用户 {df['user_id'].nunique():,}；商品 {df['product_id'].nunique():,}")
    return df


def time_split(df):
    print("[3] 时间切分（每位用户最近一条评分留作测试）...")
    df = df.sort_values(["user_id", "last_rating_date", "product_id"], kind="stable").reset_index(drop=True)
    test_idx = df.groupby("user_id", sort=False).tail(1).index
    train = df.drop(test_idx).reset_index(drop=True)
    test = df.loc[test_idx].reset_index(drop=True)
    train.to_csv(os.path.join(PREP, "train.csv"), index=False, encoding="utf-8-sig")
    test.to_csv(os.path.join(PREP, "test.csv"), index=False, encoding="utf-8-sig")
    print(f"[切分] train {len(train):,}；test {len(test):,}")
    return train, test


def cv_evaluate(df, reader):
    print("[4] 模型对比（5 折交叉验证）...")
    top_items = df.groupby("product_id", sort=False)["user_id"].size().nlargest(TOP_PRODUCTS_FOR_CV).index
    sub = df[df["product_id"].isin(top_items)].sample(CV_SAMPLE_SIZE, random_state=42)
    data = Dataset.load_from_df(sub[["user_id", "product_id", "rating"]], reader)

    rows = []
    for name, algo in [
        ("BaselineOnly", BaselineOnly()),
        ("SVD", SVD(n_factors=20, n_epochs=20, random_state=42)),
        ("KNNBasic(Item)", KNNBasic(k=40, min_k=1,
                                    sim_options={"name": "cosine", "user_based": False},
                                    random_state=42)),
    ]:
        t0 = time.time()
        res = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=False)
        rows.append({
            "模型": name,
            "评估数据": f"热门Top{TOP_PRODUCTS_FOR_CV}抽样{len(sub):,}条",
            "RMSE": round(float(np.mean(res["test_rmse"])), 4),
            "RMSE_std": round(float(np.std(res["test_rmse"])), 4),
            "MAE": round(float(np.mean(res["test_mae"])), 4),
            "MAE_std": round(float(np.std(res["test_mae"])), 4),
            "耗时秒": round(time.time() - t0, 1),
        })
        print(f"  {name}: RMSE={rows[-1]['RMSE']} MAE={rows[-1]['MAE']} ({rows[-1]['耗时秒']}s)")

    # KNNBasic(User) 内存随用户数平方增长，用小规模用户抽样做对照
    users5k = df["user_id"].drop_duplicates().sample(USER_KNN_SAMPLE, random_state=42)
    sub_u = df[df["user_id"].isin(users5k)]
    data_u = Dataset.load_from_df(sub_u[["user_id", "product_id", "rating"]], reader)
    t0 = time.time()
    res = cross_validate(
        KNNBasic(k=40, min_k=1, sim_options={"name": "cosine", "user_based": True}, random_state=42),
        data_u, measures=["RMSE", "MAE"], cv=5, verbose=False,
    )
    rows.append({
        "模型": "KNNBasic(User)",
        "评估数据": f"{USER_KNN_SAMPLE}用户抽样({len(sub_u):,}条)",
        "RMSE": round(float(np.mean(res["test_rmse"])), 4),
        "RMSE_std": round(float(np.std(res["test_rmse"])), 4),
        "MAE": round(float(np.mean(res["test_mae"])), 4),
        "MAE_std": round(float(np.std(res["test_mae"])), 4),
        "耗时秒": round(time.time() - t0, 1),
    })
    print(f"  KNNBasic(User): RMSE={rows[-1]['RMSE']} MAE={rows[-1]['MAE']} ({rows[-1]['耗时秒']}s)")
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(EVAL, "model_comparison.csv"), index=False, encoding="utf-8-sig")
    return comp


def topn_evaluate(train, test, reader):
    print("[5] Top-N 评估（SVD / SVD+热门先验 / 热门基线，全商品宇宙）...")
    data_full = Dataset.load_from_df(train[["user_id", "product_id", "rating"]], reader)
    trainset = data_full.build_full_trainset()
    t0 = time.time()
    svd = SVD(n_factors=20, n_epochs=20, random_state=42)
    svd.fit(trainset)
    print(f"  SVD 全量训练完成（{time.time()-t0:.1f}s）")

    pop = train.groupby("product_id", sort=False)["user_id"].size()
    all_items = train["product_id"].drop_duplicates().sort_values().reset_index(drop=True)
    M = len(all_items)
    inner_iid = np.array([trainset.to_inner_iid(p) for p in all_items], dtype=np.int64)
    pop_all = pop.reindex(all_items).values.astype(float)
    test_users = test["user_id"].drop_duplicates().sample(TOP_N_TEST_USERS, random_state=42)
    inner_uid = np.array(
        [trainset.to_inner_uid(u) for u in test_users if u in trainset._raw2inner_id_users],
        dtype=np.int64,
    )
    test_map = test.set_index("user_id")["product_id"].to_dict()
    g = svd.trainset.global_mean

    # 用户已购商品（训练内）集合，用于排除重复推荐
    user_train = {}
    for row in train[train["user_id"].isin(test_users)].itertuples(index=False):
        user_train.setdefault(row.user_id, set()).add(row.product_id)

    def eval_model(score_fn):
        hits = {k: 0 for k in KS}
        rec_items = set()
        for start in range(0, len(inner_uid), EVAL_CHUNK):
            end = min(start + EVAL_CHUNK, len(inner_uid))
            batch_uid = inner_uid[start:end]
            scores = score_fn(batch_uid)
            for ii, uid in enumerate(batch_uid):
                for p in user_train.get(test_users.iloc[start + ii], ()):
                    j = np.searchsorted(all_items, p)
                    if j < M and all_items[j] == p:
                        scores[ii, j] = -np.inf
            for k in KS:
                # argpartition(kth=M-k) 保证 M-k 位置之后为最大的 k 个值
                topk = np.argpartition(scores, kth=M - k, axis=1)[:, M - k:]
                for ii, uid in enumerate(batch_uid):
                    tp = test_map.get(test_users.iloc[start + ii])
                    if tp is None:
                        continue
                    j = np.searchsorted(all_items, tp)
                    if j < M and all_items[j] == tp and j in topk[ii]:
                        hits[k] += 1
                for idx in topk.ravel():
                    rec_items.add(all_items[idx])
        return hits, rec_items

    svd_fn = lambda b: g + svd.bu[b][:, None] + svd.bi[inner_iid][None, :] + svd.pu[b] @ svd.qi[inner_iid].T
    svd_hits, svd_rec = eval_model(svd_fn)
    pop_hits, pop_rec = eval_model(lambda b: np.tile(pop_all[None, :], (len(b), 1)))
    n = len(inner_uid)

    # 热门先验 λ 网格搜索（以 Recall@10 为准则，同一样本快速评估）
    def recall10(score_fn):
        hits = 0
        for start in range(0, len(inner_uid), EVAL_CHUNK):
            end = min(start + EVAL_CHUNK, len(inner_uid))
            batch_uid = inner_uid[start:end]
            scores = score_fn(batch_uid)
            for ii, uid in enumerate(batch_uid):
                for p in user_train.get(test_users.iloc[start + ii], ()):
                    j = np.searchsorted(all_items, p)
                    if j < M and all_items[j] == p:
                        scores[ii, j] = -np.inf
            topk = np.argpartition(scores, kth=M - 10, axis=1)[:, M - 10:]
            for ii, uid in enumerate(batch_uid):
                tp = test_map.get(test_users.iloc[start + ii])
                if tp is None:
                    continue
                j = np.searchsorted(all_items, tp)
                if j < M and all_items[j] == tp and j in topk[ii]:
                    hits += 1
        return hits / n

    grid = {}
    for lam in (0.25, 0.5, 1.0, 2.0):
        grid[lam] = recall10(lambda b, lam=lam: svd_fn(b) + lam * np.log1p(pop_all)[None, :])
    best_lam = max(grid, key=grid.get)
    print(f"  热门先验 λ 网格: {grid} → 选 λ={best_lam}")
    hybrid_hits, hybrid_rec = eval_model(
        lambda b: svd_fn(b) + best_lam * np.log1p(pop_all)[None, :]
    )

    rows = []
    for k in KS:
        for model, hits, rec in (
            ("SVD", svd_hits, svd_rec),
            ("SVD+热门先验", hybrid_hits, hybrid_rec),
            ("热门基线", pop_hits, pop_rec),
        ):
            p = hits[k] / (n * k)
            r = hits[k] / n
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0
            cov = len(rec) / M
            rows.append({"模型": model, "k": k,
                         "Precision@k": round(p, 4), "Recall@k": round(r, 4),
                         "F1@k": round(f1, 4), "覆盖率": round(cov, 4)})
    ev = pd.DataFrame(rows)
    ev.to_csv(os.path.join(EVAL, "topn_evaluation.csv"), index=False, encoding="utf-8-sig")
    print(ev.to_string(index=False))
    return ev, svd, trainset, all_items, inner_iid, inner_uid, test_users, test_map, pop, best_lam


def sample_recommendations(svd, trainset, all_items, inner_iid, inner_uid, test_users, test_map, k=10):
    print("[6] 生成示例推荐 ...")
    train_items = {}
    for row in pd.read_csv(
        os.path.join(PREP, "train.csv"), encoding="utf-8-sig",
        dtype={"user_id": str, "product_id": str}, usecols=["user_id", "product_id"],
    ).itertuples(index=False):
        train_items.setdefault(row.user_id, set()).add(row.product_id)
    rows = []
    g = svd.trainset.global_mean
    for i in range(min(10, len(inner_uid))):
        u = test_users.iloc[i]
        uid = inner_uid[i]
        score = g + svd.bu[uid] + svd.bi[inner_iid] + svd.pu[uid] @ svd.qi[inner_iid].T
        for p in train_items.get(u, ()):
            j = np.searchsorted(all_items, p)
            if j < len(all_items) and all_items[j] == p:
                score[j] = -np.inf
        order = np.argsort(score)[::-1][:k]
        for pos, j in enumerate(order, 1):
            pid = all_items[j]
            rows.append({
                "用户": u, "推荐位次": pos, "商品": pid,
                "预测分": round(float(score[j]), 3), "是否命中测试": pid == test_map.get(u),
            })
    pd.DataFrame(rows).to_csv(os.path.join(EVAL, "recommendations_sample.csv"),
                              index=False, encoding="utf-8-sig")


def cold_start_outputs(train, pop):
    print("[7] 冷启动输出 ...")
    prod_meta = pd.read_csv(
        os.path.join(BASE, "..", "..", "01_data_engineering", "task2_clean", "output", "amazon_clean.csv"),
        encoding="utf-8-sig", usecols=["product_id", "category_l1", "rating"],
    )
    pop_df = pop.rename("rating_count").nlargest(50).reset_index()
    pop_df = pop_df.merge(prod_meta, on="product_id", how="left")
    pop_df.to_csv(os.path.join(COLD, "popularity_baseline.csv"), index=False, encoding="utf-8-sig")

    # 新品冷启动示例：取 3 个热门商品模拟“新上架”，给出同品类热门榜
    examples = []
    sample_new = train["product_id"].drop_duplicates().sample(3, random_state=1)
    for pid in sample_new:
        cat = prod_meta.loc[prod_meta["product_id"] == pid, "category_l1"]
        if cat.empty or pd.isna(cat.iloc[0]):
            rec = pop.nlargest(5).index.tolist()
            note = "无品类信息，回退热门榜"
        else:
            same_cat = prod_meta[prod_meta["category_l1"] == cat.iloc[0]]["product_id"]
            rec = pop[same_cat[same_cat.isin(pop.index)]].nlargest(5).index.tolist()
            note = f"同品类（{cat.iloc[0]}）热门榜"
        examples.append({"新品": pid, "推荐策略": note, "推荐商品": ";".join(map(str, rec))})
    pd.DataFrame(examples).to_csv(os.path.join(COLD, "new_item_examples.csv"),
                                  index=False, encoding="utf-8-sig")


def charts(comp, ev):
    print("[8] 图表 ...")
    fig, ax = plt.subplots(figsize=(9, 5))
    names = comp["模型"].tolist()
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w / 2, comp["RMSE"], w, label="RMSE", color="#2E74B5")
    ax.bar(x + w / 2, comp["MAE"], w, label="MAE", color="#ED7D31")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_title("评分预测模型对比（5 折交叉验证）")
    ax.legend()
    for i in range(len(names)):
        ax.text(i - w / 2, comp["RMSE"][i], f"{comp['RMSE'][i]:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, comp["MAE"][i], f"{comp['MAE'][i]:.3f}", ha="center", va="bottom", fontsize=8)
    save_fig(fig, "model_rmse_mae.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in ((axes[0], "Precision@k"), (axes[1], "Recall@k")):
        for model, color in (("SVD", "#2E74B5"), ("热门基线", "#BFBFBF")):
            sub = ev[ev["模型"] == model]
            ax.plot(sub["k"], sub[metric], marker="o", label=model, color=color)
        ax.set_xlabel("k"); ax.set_ylabel(metric); ax.set_title(f"{metric} 随 k 变化")
        ax.legend()
    save_fig(fig, "precision_recall_k.png")


def main():
    print("=" * 70)
    print("任务2：构建推荐模型（协同过滤）")
    print("=" * 70)
    up = load_data()
    df = filter_subset(up)
    train, test = time_split(df)
    reader = Reader(rating_scale=(1, 5))
    comp = cv_evaluate(df, reader)
    ev, svd, trainset, all_items, inner_iid, inner_uid, test_users, test_map, pop, best_lam = topn_evaluate(train, test, reader)
    sample_recommendations(svd, trainset, all_items, inner_iid, inner_uid, test_users, test_map)
    cold_start_outputs(train, pop)
    charts(comp, ev)

    # 数据规模与稀疏度摘要（打印）
    per_user = df.groupby("user_id", sort=False)["product_id"].size()
    sparsity = len(df) / (df["user_id"].nunique() * df["product_id"].nunique()) * 100
    print("\n[摘要]")
    print(f"  过滤后评分 {len(df):,} 条；用户 {df['user_id'].nunique():,}；商品 {df['product_id'].nunique():,}")
    print(f"  人均评分（均值/中位数）：{per_user.mean():.2f} / {int(per_user.median())}；"
          f"稀疏度 {sparsity:.5f}%")
    print(f"  train {len(train):,} / test {len(test):,}（时间切分，每位用户最近一条留出）")
    best_cv = comp.loc[comp["MAE"].idxmin()]
    print(f"  评分预测最优：{best_cv['模型']} MAE={best_cv['MAE']:.4f}（5 折 CV）")
    print(f"  热门先验最优 λ={best_lam}；Top-N 结果见 topn_evaluation.csv")
    print("[完成] 输出已写入 output/")


if __name__ == "__main__":
    main()
