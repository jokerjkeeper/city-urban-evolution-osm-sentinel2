import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
from PIL import Image

# ==========================================
# 3. 訓練「繁榮度預測器」
# ==========================================
class ProsperityPredictor:
    def __init__(self):
        # 使用隨機森林回歸，適合處理 2048 維的視覺特徵向量
        self.regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    def train(self, df):
        # 1. 構建「訓練集」與「測試集」
        # X: ResNet 提取的視覺向量, y: OSM 建築密度
        X = np.stack(df['features'].values)
        y = df['building_count'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"正在訓練模型... 樣本數: {len(X_train)}")
        self.regressor.fit(X_train, y_train)

        # 評估模型
        score = self.regressor.score(X_test, y_test)
        print(f"模型訓練完成，R² 分數: {score:.4f}")
        return score

    def predict(self, features):
        return self.regressor.predict(features.reshape(1, -1))[0]

# ==========================================
# 4. 比較繁榮度分數 (Change Analysis)
# ==========================================
def analyze_prosperity_change(model, resnet, img_2018_dir, img_2024_dir):
    """
    比較同個地點在不同年份的預測分數。
    計算公式：$$\Delta S = S_{2024} - S_{2018}$$
    """
    results = []
    # 這裡假設兩個資料夾內有相同檔名的座標圖
    for filename in os.listdir(img_2024_dir):
        path_18 = os.path.join(img_2018_dir, filename)
        path_24 = os.path.join(img_2024_dir, filename)

        if os.path.exists(path_18) and os.path.exists(path_24):
            feat_18 = resnet.get_vector(path_18)
            feat_24 = resnet.get_vector(path_24)

            s_18 = model.predict(feat_18)
            s_24 = model.predict(feat_24)

            results.append({
                "location": filename,
                "score_2018": s_18,
                "score_2024": s_24,
                "growth": s_24 - s_18
            })
    return pd.DataFrame(results)


def cluster_city_development(osm_gdf, n_clusters=8):
    """
    使用 K-Means 將建築物進行分群分析
    """
    # 1. 提取座標 (經度, 緯度)
    coords = np.array([(geom.centroid.x, geom.centroid.y) for geom in osm_gdf.geometry])

    # 2. 執行 K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    osm_gdf['cluster_label'] = kmeans.fit_predict(coords)

    # 3. 繪製結果
    fig, ax = plt.subplots(figsize=(12, 10))
    # 根據群組編號上色
    osm_gdf.plot(column='cluster_label', ax=ax, cmap='Set3', markersize=5, legend=True)

    # 畫出群聚中心 (代表該區的核心)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='重心')

    plt.title(f"台中城市結構群聚分析 (K={n_clusters})")
    plt.legend()
    plt.show()

    return kmeans.cluster_centers_



# ==========================================
# 功能 2：ResNet 模型對接與特徵提取
# ==========================================
class CityResNet:
    def __init__(self):
        # 使用 ResNet50 (更深層，特徵更豐富，適合城市場景分析)
        self.model = models.resnet50(pretrained=True)
        
        # 保留多層特徵：提取中間層和最後層的特徵
        # 這樣可以同時獲取低級特徵(邊緣、紋理)和高級特徵(語義信息)
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

        # 增強預處理：添加數據增強以提高魯棒性
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),  # 更大的輸入尺寸保留更多細節
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_vector(self, img_path):
        """
        將圖片轉為 2048 維度的特徵向量 (ResNet50)
        
        注意：ResNet 預訓練於 ImageNet，擅長識別物體，但對於城市場景的「鬧區」特徵
        需要結合以下方法提升效果：
        1. 使用更大的模型 (ResNet50 vs ResNet18)
        2. 結合 OSM 數據的空間特徵 (建築密度、POI 數量等)
        3. 考慮使用城市場景專用的預訓練模型 (如 PlacesCNN)
        4. 進行遷移學習/微調
        """
        try:
            img = Image.open(img_path).convert('RGB')
            
            # 檢查圖片是否有效
            if img.size[0] < 10 or img.size[1] < 10:
                print(f"警告：圖片尺寸過小 {img_path}")
                return None
            
            tensor = self.preprocess(img).unsqueeze(0)
            with torch.no_grad():
                vector = self.feature_extractor(tensor).flatten().numpy()
            
            # 返回 2048 維特徵向量
            return vector
        except Exception as e:
            print(f"特徵提取失敗 {img_path}: {e}")
            return None
        