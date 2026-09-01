# Amazon 电商数据商业化分析项目
基于 Kaggle 公开数据集（Amazon Product Reviews 与 Amazon Sales Dataset）的端到端商业数据分析项目。以“数据 → 洞察 → 行动”为主线，覆盖**数据工程、探索性分析与可视化、建模与深度洞察、商业应用**四个阶段，最终输出可直接用于业务决策的用户分层、推荐原型与品类经营建议。

## 项目背景

在缺少订单、浏览、点击日志的有限数据条件下，本项目以**评分与评论行为为代理信号**，回答电商经营中最常见的三类问题：卖什么（品类结构）、卖给谁（用户分层）、怎么卖（推荐与营销策略），并对结论的数据口径与局限做显式声明，避免“为结论造数据”。

## 核心业务问题

1. 哪些品类“增长快但差评较多”，需要优先优化？
2. 高价值用户是谁、他们在评论中最关注什么？
3. 如何基于热度代理做库存分级建议？

## 主要发现

- **数据规模**：782 万条评分、420 万用户、1,351 个商品、10,598 条有效评论。
- **品类诊断**：Electronics（负面占比 8.3%）与 Home&Kitchen（8.0%）属于“高热度 + 高差评”，列为优先优化品类；Computers&Accessories 为热销关注品类。
- **差评主题**：LDA 提炼出 8 个主题，差评集中在电视与画质（9.6%）、充电与线材（9.2%）、影音与音质（8.9%），可作为产品改进优先级。
- **用户分层**：高价值用户约 84 万人（20%），人均评分 4.7 次、活跃跨度约 779 天、100% 跨品类，适合会员运营与跨品类交叉推荐。
- **推荐模型**：SVD 评分预测 MAE 0.792（5 折交叉验证）；加入热门先验后 Recall@20 ≈ 3.9%，与热门基线相当，但覆盖率（1.53%）显著更高，能挖掘长尾商品。
- **关联规则**：114 条“买了又买 / 看了又看”规则，提升度最高约 63，可直接用于商品详情页推荐位。

## 技术栈

Python 3 · Pandas · NumPy · Matplotlib · Scikit-learn · SciPy · NLTK · TextBlob · WordCloud · Gensim · scikit-surprise

## 方法概览（四阶段）

| 阶段 | 目录 | 核心内容 |
|---|---|---|
| ① 数据工程 | `01_data_engineering/` | 数据探索与字典、清洗与评论长表展开、业务指标与三张宽表 |
| ② 探索性分析与可视化 | `02_eda_visualization/` | 销售/用户/评论分析、规则 + KMeans 用户分层、Tableau 仪表盘输入表 |
| ③ 建模与深度洞察 | `03_modeling/` | TextBlob + VADER 情感分析、LDA 主题建模、协同过滤、物品关联规则 |
| ④ 商业应用 | `04_business_analysis/` | 品类 × 热度 × 差评矩阵、高价值用户 × 评论主题整合、库存分级建议 |

## 目录结构

```text
.
├── README.md
├── requirements.txt
├── 01_data_engineering/
│   ├── task1_explore/01_explore_and_dict.py
│   ├── task2_clean/02_clean_and_integrate.py
│   └── task3_metrics/03_build_metrics_and_wide_tables.py
├── 02_eda_visualization/
│   ├── task1_user_sales/11_user_sales_analysis.py
│   ├── task2_product_review/12_product_review_analysis.py
│   └── task3_dashboard/
│       ├── 13_dashboard_data_prep.py      # Tableau 仪表盘输入表导出
│       └── dashboard.twbx                 # Tableau 交互式仪表盘工作簿（含数据）
├── 03_modeling/
│   ├── task1_sentiment_lda/21_review_sentiment_lda.py
│   └── task2_recommendation/
│       ├── 22_recommendation_cf.py
│       └── 23_association_rules.py
└── 04_business_analysis/
    └── 04_business_analysis.py
```

## 数据说明

原始数据来自 Kaggle，**不随仓库上传**，请自行下载后放入工程根目录：

| 文件 | 内容 | 来源 |
|---|---|---|
| `amazon.csv` | 商品属性表（16 字段） | Amazon Sales Dataset |
| `ratings_electronics.csv` | 用户评分行为表（约 782 万条，无表头） | Amazon Product Reviews |

Kaggle 原始文件名为 `ratings_Electronics (1).csv`（含空格和括号），建议重命名为 `ratings_electronics.csv`；脚本会自动兼容两种文件名。数据放在其他位置时，可通过环境变量指定：

```powershell
$env:AMAZON_DATA_DIR = "D:\data\amazon"
python 01_data_engineering\task1_explore\01_explore_and_dict.py
```

## 快速开始

```bash
pip install -r requirements.txt
```

按流水线顺序执行（每个脚本的输出是下一个脚本的输入）：

```powershell
# ① 数据工程
python 01_data_engineering\task1_explore\01_explore_and_dict.py
python 01_data_engineering\task2_clean\02_clean_and_integrate.py
python 01_data_engineering\task3_metrics\03_build_metrics_and_wide_tables.py

# ② 探索性分析与可视化
python 02_eda_visualization\task1_user_sales\11_user_sales_analysis.py
python 02_eda_visualization\task2_product_review\12_product_review_analysis.py
python 02_eda_visualization\task3_dashboard\13_dashboard_data_prep.py

# ③ 建模与深度洞察
python 03_modeling\task1_sentiment_lda\21_review_sentiment_lda.py
python 03_modeling\task2_recommendation\22_recommendation_cf.py
python 03_modeling\task2_recommendation\23_association_rules.py

# ④ 商业分析
python 04_business_analysis\04_business_analysis.py
```

各脚本产物（清洗后数据、宽表、图表、模型评估结果）写入各自任务目录下的 `output/`，全部可一键复现，因此不随仓库提交。

## 可视化成果

- **Tableau 交互式仪表盘**：工作簿见 [dashboard.twbx](02_eda_visualization/task3_dashboard/dashboard.twbx)（已打包数据，下载后用 Tableau Desktop / Tableau Reader 打开即可查看）。

## 口径与局限

- **GMV（估算）**：折后价 × 评分数量，为销量代理口径（无订单表）。
- **复购率**：以“重复评分”代理“重复购买”；数据中同一用户对同一商品仅一次评分，同商品复购恒为 0，采用跨商品复购口径。
- **热度/增长**：以评分数量代理热销与规模（无订单日期，无法计算真实增速）。
- **推荐**：评分 ≠ 购买，推荐以“共同评分”为代理；离线命中率受稀疏矩阵限制。
- 高价值用户（ratings 侧）与评论主题（amazon 侧）无法逐条关联，采用模块化整合并如实声明。

## 致谢

数据来源：[Amazon Product Reviews](https://www.kaggle.com/datasets/sreenathvadlamudi/amazon-product-reviews) 与 [Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)（Kaggle），本项目仅用于学习与研究。
