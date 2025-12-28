# 1. 创建环境
conda create -n xly python=3.11

# 2. 激活环境
conda activate xly

# 3. 安装必要的库
<!-- detectron2需要numpy<2，建议到kaggle或者有gpu的机器上安装 -->
pip install opencv-python matplotlib torch torchvision torchaudio numpy<2
pip install 'git+https://github.com/facebookresearch/detectron2.git'
