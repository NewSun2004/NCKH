import pandas as pd

import joblib
import os

from loadData import DataLoader


class HyBridRecommender():
    def __init__(self, meta_data, CF_data, CBF_trainset):
        self.meta_data = meta_data
        self.CF_data = CF_data
        self.CBF_trainset = CBF_trainset

        # Tải model KNN_CBF đã lưu
        model_file = "\\Source_code\\3. Hệ thống đề xuất\\Best recommendation models\\KNN_CBF.pkl"
        self.KNN_CBF = joblib.load(model_file)
        
        # Tải model SVD_CF đã lưu
        model_file = "\\Source_code\\3. Hệ thống đề xuất\\Best recommendation models\\SVD_CF.pkl"
        self.SVD_CF = joblib.load(model_file)

    def get_content_based_recommendations(self, itemID, top_n) -> list:
        # Truy xuất item features từ mã asin
        itemFeatures = self.meta_data.loc[self.meta_data['asin'] == itemID][["encodedCategory", "vectorizedTitle", "encodedBrand"]]

        # Tìm k sản phẩm liên quan đến asin được nhập
        distances, neighbors = self.KNN_CBF.kneighbors(itemFeatures)

        # Xuất sản phẩm được đề xuất và xóa sản phẩm được nhập
        recommendedItemID = pd.DataFrame(self.meta_data.iloc[neighbors.flatten()].asin.unique()[:top_n], columns=["asin"])
        recommendedItemID.drop(recommendedItemID[recommendedItemID.asin == itemID].index, inplace=True)

        return recommendedItemID.asin.tolist()

    def get_collaborative_filtering_recommendations(self, user_id, top_n=10):
        # Tạo testset cho người dùng cụ thể
        testset = self.CBF_trainset.build_anti_testset()
        testset = list(filter(lambda x: x[0] == user_id, testset))
        predictions = self.SVD_CF.test(testset)
        predictions.sort(key=lambda x: x.est, reverse=True)
        return [pred.iid for pred in predictions[:top_n]]

    def get_hybrid_recommendations(self, user_id, product_id, top_n):
        # Lấy danh sách đề xuất từ phương pháp content-based và collaborative filtering
        content_based_recommendations = self.get_content_based_recommendations(product_id, top_n)
        collaborative_filtering_recommendations = self.get_collaborative_filtering_recommendations(user_id, top_n)

        # Tạo danh sách kết quả
        hybrid_recommendations = []

        # Thêm các phần tử trùng nhau vào danh sách kết quả
        for item in content_based_recommendations:
            if item in collaborative_filtering_recommendations:
                hybrid_recommendations.append(item)

        # Đếm số lượng phần tử đã thêm vào danh sách kết quả
        num_added = len(hybrid_recommendations)

        # Nếu danh sách kết quả chưa đủ top_n, thêm xen kẽ từ cả hai nguồn
        while num_added < top_n:

            # Thêm từ collaborative filtering
            if len(collaborative_filtering_recommendations) > 0:
                hybrid_recommendations.append(collaborative_filtering_recommendations.pop(0))
                num_added += 1
                if num_added == top_n:
                    break


            # Thêm từ content-based
            if len(content_based_recommendations) > 0:
                hybrid_recommendations.append(content_based_recommendations.pop(0))
                num_added += 1
                if num_added == top_n:
                    break

        # Lọc ra các hàng có mã asin trong mảng
        product_RS = meta_data[meta_data['asin'] == product_id]
        product_RS.reset_index(inplace=True)
        product_RS = product_RS[['category', 'title', 'brand', 'asin']]

        tada = meta_data[meta_data['asin'].isin(hybrid_recommendations)].drop_duplicates()
        tada.reset_index(inplace=True)
        return tada[['category', 'title', 'brand', 'asin']]
    

# Tìm đường dẫn thư mục con và thư mục chính
base_dir = os.path.abspath(os.getcwd())

CBFDataPath = "\\Source_code\\Dữ liệu đã được xử lý\\CBF_data.csv"
CFDataPath = "\\Source_code\\Dữ liệu đã được xử lý\\CF_data.csv"

dataLoader = DataLoader(CBFDataPath, CFDataPath)
meta_data = dataLoader.meta_data
CF_data = dataLoader.CF_data
CBF_trainset = dataLoader.getCBFTrainSet()

hybridSystem = HyBridRecommender(meta_data, CF_data, CBF_trainset)

user_id = 'A2GX6DZPHMW9BQ'
product_id = 'B00004UE29'
top_n = 10
print(hybridSystem.get_hybrid_recommendations(user_id, product_id, top_n))