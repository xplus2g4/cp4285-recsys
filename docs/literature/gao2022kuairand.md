# KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos

- **Authors:** Chongming Gao\*, Shijun Li\*, Yuan Zhang\* (equal contribution), Jiawei Chen, Biao Li, Wenqiang Lei, Peng Jiang, Xiangnan He (USTC, Kuaishou, Zhejiang U., Sichuan U.)
- **Venue / year:** CIKM '22 (31st ACM International Conference on Information and Knowledge Management), Atlanta, Oct 17–21, 2022; resource/short paper, pp. 3953–3957
- **Link:** https://arxiv.org/abs/2208.08696 (DOI https://doi.org/10.1145/3511808.3557624)
- **Local PDF:** `docs/literature/papers/gao2022kuairand.pdf` (arXiv v2, 23 Aug 2022)
- **Source basis:** full text + official GitHub README (https://github.com/chongminggao/KuaiRand, `README.md` read 2026-10-14; kuairand.com points to the same content)

## Problem
Logged recommendation data is missing-not-at-random: what users see is decided by the deployed policy (exposure bias), so offline metrics on logs say nothing reliable about un-exposed user–item pairs. Missing-at-random (MAR) test data fixes this, but existing random-exposure datasets are small (Yahoo!R3: 54,000 random ratings; Coat: 4,640) or have no user IDs or features (Yahoo!R6A/B, Open Bandit with 80 items). The authors release a large-scale dataset in which random items were inserted into a real production feed.

## Method (data collection)
Three steps (§3.2):
1. **Item pool and users.** A batch of videos was sampled, and spam such as advertisements was filtered out, leaving **7,583 items**. Over 200,000 real users were randomly selected and robots removed.
2. **Random intervention, 2022-04-22 to 2022-05-08.** Each time the system recommended a list to a user, it decided with a **fixed probability (value withheld "for commercial security")** whether to intervene. If it did, it **replaced one randomly chosen video in the list** with an item **sampled uniformly from the 7,583-item pool**. 12 feedback signals were recorded. Users with fewer than 10 random exposures were removed, leaving **27,285 users**. Every one of the 7,583 items was inserted at least once, giving **1,186,059 random interventions** in total. The README adds that about **0.37%** of interactions were replaced and that "each recommended video … has an equal probability of being replaced".
3. **History.** Normal (non-intervened) logs for the same users were added for 2022-04-08 to 2022-04-21, plus user and item side information. The paper describes this as "one-month" data: **32,038,725 items** and **322,278,385 normal recommendations**.

The two-week random window **overlaps exactly with the second standard window** (`log_standard_4_22_to_5_08`). Random items appear inside the normal feed, interleaved with policy-chosen items. Policy logs come from **15 scenario-specific policies** (`tab` ∈ [0,14]), e.g. a two-column UI where the user clicks a thumbnail and a single-column UI where videos autoplay.

## Versions and counts (Table 1 and README, identical)
| Version | Log | #Users | #Items | #Interactions |
|---|---|---|---|---|
| 27K | 15 policies | 27,285 | 32,038,725 | 322,278,385 |
| 27K | random | 27,285 | 7,583 | 1,186,059 |
| 1K | 15 policies | 1,000 | 4,369,953 | 11,713,045 |
| 1K | random | 1,000 | 7,388 | 43,028 |
| Pure | 15 policies | 27,285 | 7,551 | 1,436,609 |
| Pure | random | 27,285 | 7,583 | 1,186,059 |

All versions have 30 user features and 62 item features, 12 feedback signals and timestamps. Sizes: 27K is 23 GB of logs + 23 GB of features; 1K is 829 MB + 3.5 GB; Pure is 184 MB + 10 MB. **Pure keeps only the logs for videos in the candidate pool** (the appendix says "7,582 videos" and the README says "7583"; the IDs run from 0 to 7,582). The authors recommend Pure for "debiasing in collaborative filtering models, domain adaptation, multi-task" work and note that its sequential logs are incomplete. User and video indices are shared across the three versions.

**Pure files (README):** `log_random_4_22_to_5_08_pure.csv` (83 MB), `log_standard_4_08_to_4_21_pure.csv` (80 MB), `log_standard_4_22_to_5_08_pure.csv` (21 MB), `user_features_pure.csv`, `video_features_basic_pure.csv`, `video_features_statistic_pure.csv`.

## Field definitions (README)
- **Log fields:** `user_id`, `video_id`, `date`, `hourmin`, `time_ms`, `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`, `long_view`, `play_time_ms`, `duration_ms`, `profile_stay_time`, `comment_stay_time`, `is_profile_enter`, `is_rand`, `tab`.
  - **`is_click`:** a real click in the two-column UI. In the single-column UI it means *valid_play*: `play_time_ms >= duration_ms` if `duration_ms <= 7,000`, else `play_time_ms > 7,000`.
  - **`long_view`:** `play_time_ms >= duration_ms` if `duration_ms <= 18,000`, else `play_time_ms >= 18,000`.
  - **`is_rand`:** 1 if the item came from the random intervention.
- **User features:** `user_active_degree` ∈ {high_active, full_active, middle_active, UNKNOWN}, `is_lowactive_period`, `is_live_streamer`, `is_video_author`, follow/fans/friend counts and ranges, `register_days(_range)`, and `onehot_feat0–17` (encrypted).
- **Video basic features:** `author_id`, `video_type` (NORMAL/AD), `upload_dt`, `upload_type`, `visible_status`, `video_duration` (ms), `server_width`/`server_height`, `music_id`, `music_type`, `tag`.
- **Video statistic features:** daily per-scenario *averages over one month*, e.g. `show_cnt`, `play_cnt`, `complete_play_cnt`, `valid_play_cnt`, `long_time_play_cnt`, `short_time_play_cnt`, `play_progress`, `like_cnt`, `comment_cnt`.

## Assumptions held fixed
- The random log is treated as MAR **conditional on the user being in the feed at that moment**. Users, sessions, timing, the UI scenario and the surrounding policy-chosen items are all inherited from the production policy. Only the identity of one slot is randomised.
- The item pool is a fixed, filtered set of 7,583 items, not the full catalogue. "Unbiased" holds only relative to this pool.
- Users were filtered to those with ≥10 random exposures, which conditions on activity.
- The insertion probability is assumed constant but is not disclosed.

## Experimental setup
None. This is a resource paper with no benchmark models or metrics. The only statistics are Table 1 and Fig. 3, which show the per-user distribution of normal interactions (mostly thousands) and of intervened interactions (≥10 per user).

## Key results
- Of the datasets compared, it is the largest with random exposure and real user IDs: 1,186,059 random interactions, 27,285 users and 7,583 items.
- Fig. 3: 25.0% of users have 0–4k normal interactions and 55.1% have 10–25 intervened interactions. Per-user random logs are therefore short.
- Suggested uses: debiasing, OPE, RL/interactive recommendation, long-sequence modelling, multi-task learning.

## Reported limitations and trade-offs
- The insertion probability is hidden.
- Pure drops all non-pool videos, so its sequences are incomplete.
- The 27K version is too large for many groups.
- The README notes that statistic features are averaged over one month, so they span both windows. This is leakage-relevant [INFERENCE: the authors do not call it leakage].
- The authors do not discuss residual bias in the random log (position in the feed, user self-selection of continuing to watch). Those caveats come from this project.

## Relevance to this project
- **§2 Data/splits:** confirms the file names, `is_rand`, the 7,583-item pool and the exact overlap of the random window with `log_standard_4_22_to_5_08`. The temporal guard (train on `log_standard_4_08_to_4_21`) is well founded.
- **§2 Sanity checks:** sampling was uniform, so the item-frequency distribution in the random log should be near-uniform over the pool (1,186,059 / 7,583 ≈ 156 per item on average [INFERENCE: arithmetic]).
- **§4 Labels:** `is_click` is UI-dependent and in the single-column UI it is a duration-thresholded valid play (7 s). That makes the "Click" label partly duration-coupled, much like `long_view` (18 s).
- **§6 User groups:** `user_active_degree` has 4 levels including UNKNOWN. `video_duration` and `duration_ms` give item duration. Popularity features from the statistic table cover the test window and should not be used for training-time popularity.
- **§10 Threats:** the pool restriction and the ≥10-exposure user filter belong in the threats list.
