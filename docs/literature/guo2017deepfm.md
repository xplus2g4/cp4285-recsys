# DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

- **Authors:** Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, Xiuqiang He (Harbin Institute of Technology Shenzhen; Huawei Noah's Ark Lab)
- **Venue / year:** IJCAI 2017
- **Link:** https://arxiv.org/abs/1703.04247
- **Local PDF:** `docs/literature/papers/guo2017deepfm.pdf` (arXiv v1)
- **Source basis:** full text

## Problem
CTR prediction needs both low-order feature interactions (e.g. app category × time, order 2) and high-order ones (category × gender × age). Existing models each fall short in one way:
- LR/FTRL needs hand-crafted crosses.
- FM is in practice limited to order 2.
- FNN needs FM pre-training and captures only high-order interactions.
- PNN ignores low-order interactions.
- Wide & Deep needs an engineered "wide" input.

## Method
- **Prediction:** ŷ = sigmoid(y_FM + y_DNN).
- **FM component:** y_FM = ⟨w, x⟩ + Σ_{j1<j2} ⟨V_i, V_j⟩ x_j1 x_j2. This is the order-1 term plus order-2 inner products of latent vectors.
- **Deep component:** a feed-forward network over a^(0) = [e_1, …, e_m], the per-field embeddings. Each field is one-hot or a continuous value, and every field gets an embedding of the same size k.
- **Key design:** the FM latent vectors V are the embedding layer of the DNN, so both components share one input and one embedding. This gives end-to-end training with no pre-training and no feature engineering beyond raw features.

## Assumptions held fixed
- The training label is click (y ∈ {0,1}) on logged data, and the model fits observed CTR directly. There is no correction for exposure or position bias: the logging policy's selection is taken as given.
- The input is the raw categorical and continuous fields, and interaction structure is learned rather than specified.
- The embedding dimension and architecture are shared across components. Train and test come from the same distribution: a random 90/10 split on Criteo and a temporal split on Company.

## Experimental setup
- **Criteo:** 45M click records, 13 continuous and 26 categorical features, random 90/10 split.
- **Company:** Huawei App Store game center, about 1B records; 7 days for training and the next day for testing.
- **Metrics:** AUC and Logloss.
- **Baselines:** LR, FM, FNN, IPNN, OPNN, PNN, LR&DNN (Wide & Deep) and FM&DNN.
- **Settings on Criteo, following Qu et al.:** dropout 0.5, layers 400-400-400, Adam, ReLU (tanh for IPNN), FM latent dimension 10, FTRL for LR.

## Key results
Table 2 (AUC / Logloss):

| Model | Company AUC | Company Logloss | Criteo AUC | Criteo Logloss |
|---|---|---|---|---|
| LR | 0.8640 | 0.02648 | 0.7686 | 0.47762 |
| FM | 0.8678 | 0.02633 | 0.7892 | 0.46077 |
| PNN | 0.8672 | 0.02636 | 0.7987 | 0.45214 |
| LR&DNN | 0.8673 | 0.02634 | 0.7981 | 0.46772 |
| **DeepFM** | **0.8715** | **0.02618** | **0.8007** | **0.45083** |

- Against the second-best model, DeepFM gains more than 0.37% (Company) and 0.25% (Criteo) in AUC.
- Against LR&DNN and FM&DNN (separate embeddings), it gains more than 0.48% and 0.33% in AUC.
- Its training time is close to the most efficient of the deep models. FNN's pre-training slows it down, and PNN's inner/outer products are expensive.
- **Hyperparameter study (Company):**
  - ReLU beats tanh except for IPNN.
  - The best dropout (keep-probability) is between 0.6 and 0.9.
  - 200–400 neurons per layer works well, and moving from 400 to 800 gives no gain for DeepFM.
  - Too many layers overfit.
  - A constant-width network beats increasing, decreasing and diamond shapes.

## Reported limitations and trade-offs
- The authors suggest future work on pooling to strengthen high-order learning and on GPU-cluster training at scale.
- Small offline AUC gains are argued to matter online by citing Wide & Deep's +0.275% AUC → +3.9% online CTR. DeepFM itself reports no online test.
- No bias or debiasing analysis is given.

## Relevance to this project
- **§5 Backbone:** DeepFM is the feature-based robustness backbone. User features (`user_active_degree`, `onehot_feat*`) and video features (`video_duration`, `tag`, `music_type`) become fields with a shared embedding size k, which should equal the MF dimension per §5.
- **§5 / §6 Duration:** when `video_duration` is a feature, a naive DeepFM can learn the duration–label coupling directly. Duration-debiasing gains may therefore differ from MF. Report both, and consider an ablation without the duration feature.
- **§5 Exposure:** IPS/DR weights go into DeepFM's per-sample logloss. DeepFM is a pointwise BCE model, which matches the paper's training objective.
- **§8:** the paper's hyperparameter findings (dropout, constant 200–400 width) give a reasonable fixed tuning grid that stays within the equal-budget rule.
