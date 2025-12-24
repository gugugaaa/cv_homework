INPUT:
  D: 数据库图像
  V: 词汇表大小（例如，10k）
  Kd: 返回图像数
OUTPUT:
  vocab: 视觉词（聚类中心）
  inv_index: 倒排索引（词 -> 列表 (img_id, tfidf_weight)）
  doc_norm: 每个图像向量的L2范数（用于余弦相似度）
  sift_store (可选): 存储SIFT描述符/关键点以便几何验证

PROCEDURE BuildIndex(D, V):
  desc_all = []
  img_descs = dict()

  for each image I in D:
    kp, desc = SIFT_DetectAndDescribe(I)     // 描述符: [n_i x 128]
    img_descs[I.id] = (kp, desc)
    append desc into desc_all

  // 1) 通过聚类（k-means）学习词汇
  vocab = KMeans(desc_all, V)

  // 2) 通过量化为每张图像计算TF
  tf = dict()  // tf[img_id] 是长度为 V 的直方图
  df = zeros(V)

  for each image I in D:
    kp, desc = img_descs[I.id]
    words = Quantize(desc, vocab)           // 将每个描述符映射到词_id
    hist = Histogram(words, V)              // 词频
    tf[I.id] = hist

    for each unique word_id w in words:
      df[w] += 1

  // 3) 计算IDF
  N = |D|
  idf[w] = log((N + 1) / (df[w] + 1)) + 1   // 平滑IDF

  // 4) 构建TF-IDF向量并建立倒排索引
  inv_index = empty list for each word w
  doc_norm = dict()

  for each image I in D:
    vec = tf[I.id] * idf                    // 元素乘（逐元素）
    vec = L2_Normalize(vec)
    doc_norm[I.id] = 1                      // 已经归一化

    for each word w where vec[w] > 0:
      inv_index[w].append( (I.id, vec[w]) )

  // 5) 可选：存储关键点/描述符以便后续几何验证
  sift_store = img_descs

  return vocab, idf, inv_index, sift_store

INPUT:
  Q: 查询图像
  vocab, idf, inv_index, sift_store
  Kd: top 结果
  use_geo: 是否进行几何验证
OUTPUT:
  排序后的图像 id 列表

PROCEDURE Retrieve(Q, vocab, idf, inv_index, sift_store, Kd, use_geo):
  kp_q, desc_q = SIFT_DetectAndDescribe(Q)
  words_q = Quantize(desc_q, vocab)
  tf_q = Histogram(words_q, V)
  vec_q = tf_q * idf
  vec_q = L2_Normalize(vec_q)

  // 1) 通过倒排索引对候选项打分（稀疏向量余弦）
  score = dict default 0
  for each word w where vec_q[w] > 0:
    for (img_id, weight) in inv_index[w]:
      score[img_id] += vec_q[w] * weight

  candidates = TopM(score, M)               // 保留前 M 个以加速（例如，M=200）

  // 2) 可选：使用 SIFT 匹配 + RANSAC 进行几何验证重排序
  if use_geo:
    geo_score = dict()
    for each img_id in candidates:
      kp_d, desc_d = sift_store[img_id]

      matches = NN_Match(desc_q, desc_d)                // 最近邻匹配
      matches = LoweRatioTest(matches, ratio=0.75)      // Lowe 比率测试
      inliers = RANSAC_Homography(kp_q, kp_d, matches)  // RANSAC 估计单应性并检验几何一致性

      geo_score[img_id] = |inliers|                     // 或基于内点的加权得分

    // 合并得分（简单方式）
    final_score[img_id] = score[img_id] + alpha * geo_score[img_id]
    ranked = SortBy(final_score, descending)
  else:
    ranked = SortBy(score, descending)

  return TopK(ranked, Kd)

INPUT:
  Qset: 查询图像集合
  GT(q): 每个查询 q 的真实相关图像 id 集合
  Retrieve(): 检索函数
  K_eval: 评估到前 K（例如，100）
OUTPUT:
  mAP、平均 Precision@K、平均 Recall@K（可选）

PROCEDURE Evaluate(Qset, GT, K_eval):
  AP_list = []
  P_at_K_list = []
  R_at_K_list = []

  for each query q in Qset:
    ranked = Retrieve(q, ...)               // 完整排序或至少前 K_eval
    rankedK = ranked[1..K_eval]

    rel = 0
    precisions_at_hits = []

    for i from 1 to K_eval:
      if rankedK[i] in GT(q):
        rel += 1
        precisions_at_hits.append( rel / i )

    // 平均精度（AP）
    if |GT(q)| > 0:
      AP = Sum(precisions_at_hits) / |GT(q)|
    else:
      AP = 0
    AP_list.append(AP)

    // Precision@K 和 Recall@K
    P_at_K = (# of items in rankedK that are in GT(q)) / K_eval
    R_at_K = (# of items in rankedK that are in GT(q)) / |GT(q)|
    P_at_K_list.append(P_at_K)
    R_at_K_list.append(R_at_K)

  mAP = Mean(AP_list)
  meanP = Mean(P_at_K_list)
  meanR = Mean(R_at_K_list)

  return mAP, meanP, meanR
