# 下载对应数据集
# BraTS: https://www.synapse.org/Synapse:syn51514105  自行登录后下载数据集
# BraTs2023数据集包含：
# BraTS2023_2017_GLI_Mapping.xlsx
# ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip
# ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData.zip
# mainifest.csv
# ISIC: https://challenge.isic-archive.com/data/#2018  自行登录后下载 2018年数据集
# ISIC2018数据集包含：
# ISIC2018_Task1_Training_GroundTruth.zip
# ISIC2018_Task1-2_Training_Input.zip
# ISIC2018_Task1_Validation_GroundTruth.zip
# ISIC2018_Task1-2_Validation_Input.zip
# ISIC2018_Task1-2_Test_Input.zip
# ISIC2018_Task1_Test_GroundTruth.zip
# MSD_Liver: https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar  自行登录后下载数据集 Task03_Liver
# MSD_Liver数据集包含：
# Task03_Liver.zip
# 解压
# BraTS2023数据集解压
mkdir -p data/BraTS2023/training_data data/BraTS2023/validation_data # 已创建
unzip data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip -d data/BraTS2023/training_data # 已解压
mv data/BraTS2023/training_data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/* data/BraTS2023/training_data/ # 已完成
unzip data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData.zip -d data/BraTS2023/validation_data # 已解压
mv data/BraTS2023/validation_data/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/* data/BraTS2023/validation_data/ # 已完成

#  ISIC2018数据集解压
mkdir -p data/ISIC2018 # 已创建
unzip data/ISIC2018/ISIC2018_Task1_Training_GroundTruth.zip -d data/ISIC2018 # 已解压
unzip data/ISIC2018/ISIC2018_Task1-2_Training_Input.zip -d data/ISIC2018 # 已解压
unzip data/ISIC2018/ISIC2018_Task1_Validation_GroundTruth.zip -d data/ISIC2018 # 已解压
unzip data/ISIC2018/ISIC2018_Task1-2_Validation_Input.zip -d data/ISIC2018 # 已解压
unzip data/ISIC2018/ISIC2018_Task1-2_Test_Input.zip -d data/ISIC2018 # 已解压
unzip data/ISIC2018/ISIC2018_Task1_Test_GroundTruth.zip -d data/ISIC2018 # 已解压


mkdir -p data/MSD_Liver # 已创建
tar -xvf data/Task03_Liver.tar -C data/MSD_Liver



pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt

