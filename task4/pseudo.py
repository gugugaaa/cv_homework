训练阶段（离线）
Algorithm Train_ViolaJones
Input: 正样本集合 F, 负样本集合 N
Output: 级联分类器 Cascade

1. 对所有训练样本计算积分图 Integral Image
2. 在 24×24 窗口内生成所有 Haar-like 特征集合 H

3. 初始化 Cascade ← 空
4. while 整体误检率 > 目标误检率 do
      4.1 使用 AdaBoost 训练一个强分类器 Stage
          - 每一轮选择一个最优 Haar 特征作为弱分类器
          - 组合若干弱分类器形成强分类器
      4.2 调整 Stage 阈值，使其保持高检测率
      4.3 将 Stage 加入 Cascade
      4.4 用当前 Cascade 扫描负样本，收集误检样本作为新的负样本
   end while

5. 返回 Cascade

检测阶段（在线）

Algorithm Detect_ViolaJones
Input: 测试图像 I, 级联分类器 Cascade
Output: 人脸检测框集合 D

1. 将图像 I 转为灰度并计算积分图
2. for 各个尺度 s do
      2.1 在图像上进行滑动窗口扫描
      2.2 对每个窗口依次通过 Cascade 中的各级分类器
          - 若在某一级被拒绝，则立即丢弃该窗口
          - 若通过所有级，则判定为人脸候选
   end for
3. 对所有候选检测框进行合并（如非极大值抑制）
4. 输出最终检测结果 D

算法要点说明

积分图使 Haar 特征可在常数时间内计算

AdaBoost 同时完成特征选择与分类器训练

级联结构通过“由粗到细”的方式大幅减少计算量

多尺度滑窗用于检测不同大小的人脸