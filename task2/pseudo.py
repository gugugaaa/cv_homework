// 1. 离线索引构建

FUNCTION build_index(image_database):
    index = FLANNIndex()  // 使用近似最近邻库（如 FLANN）
    image_features = {}   // {image_id: (keypoints, descriptors)}
    
    FOR each image_path IN image_database:
        image = load_image(image_path)
        keypoints, descriptors = SIFT.detect_and_compute(image)  // 提取 SIFT 特征
        image_id = get_image_id(image_path)
        
        // 将描述子存入索引，关联 image_id
        index.add(descriptors, image_id)  
        image_features[image_id] = (keypoints, descriptors)
    
    index.build()  // 构建索引结构
    RETURN index, image_features

// 2. 在线查询处理

FUNCTION query_image(query_img, index, image_features, top_k=10):
    // 提取查询图像特征
    q_keypoints, q_descriptors = SIFT.detect_and_compute(query_img)
    
    // 搜索最近邻
    matches = index.knn_search(q_descriptors, k=2)  // 返回每个查询点的 2 个最近邻
    
    // 比率测试过滤（Lowe's ratio test）
    good_matches = []
    FOR each match_pair IN matches:
        IF match_pair[0].distance < 0.8 * match_pair[1].distance:
            good_matches.append(match_pair[0])
    
    // 统计匹配图像的投票数
    image_votes = {}  // {image_id: vote_count}
    FOR match IN good_matches:
        train_img_id = match.train_image_id  // 获取匹配到的库图像ID
        image_votes[train_img_id] = image_votes.get(train_img_id, 0) + 1
    
    // 按投票数排序并返回 top_k 结果
    ranked_results = sort(image_votes, by=value, descending=True)
    RETURN top_k images from ranked_results

// 3. 系统性能度量

FUNCTION evaluate_system(queries, ground_truth, index, image_features):
    AP_list = []  // 存储每个查询的 Average Precision
    
    FOR each query_img, relevant_set IN queries:
        // relevant_set: 人工标注的相关图像ID集合
        
        // 执行查询
        retrieved_list = query_image(query_img, index, image_features, top_k=len(image_database))
        
        // 计算 Precision-Recall 曲线
        relevant_count = 0
        precision_sum = 0
        
        FOR rank, img_id IN enumerate(retrieved_list):
            IF img_id IN relevant_set:
                relevant_count += 1
                current_precision = relevant_count / (rank + 1)
                precision_sum += current_precision
        
        // 计算当前查询的 AP
        num_relevant = len(relevant_set)
        AP = precision_sum / num_relevant IF num_relevant > 0 ELSE 0
        AP_list.append(AP)
    
    mAP = mean(AP_list)  // 系统平均精度
    RETURN mAP