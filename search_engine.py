"""
Module 3: Logic Tìm kiếm & Phân loại (Search Engine & Logic)
Xử lý logic so sánh để tìm ra sự tương đồng hoặc phân loại âm thanh.
"""

import numpy as np
from database_manager import DatabaseManager


class SearchEngine:

    
    def __init__(self, db_manager=None):

        if db_manager is None:
            self.db = DatabaseManager()
        else:
            self.db = db_manager
    
    def euclidean_distance(self, vector1, vector2):

        if len(vector1) != len(vector2):
            raise ValueError("Hai vector phải có cùng kích thước")
        
        sum_squared_diff = 0.0
        for i in range(len(vector1)):
            diff = vector1[i] - vector2[i]
            sum_squared_diff += diff * diff
        
        # Tính căn bậc 2 thủ công (Newton-Raphson method)
        distance = self._sqrt(sum_squared_diff)
        return distance
    
    def _sqrt(self, n, precision=1e-10):

        if n < 0:
            raise ValueError("Không thể tính căn bậc 2 của số âm")
        if n == 0:
            return 0
        
        x = n
        while True:
            root = 0.5 * (x + n / x)
            if abs(root - x) < precision:
                return root
            x = root
    
    def cosine_similarity(self, vector1, vector2):

        if len(vector1) != len(vector2):
            raise ValueError("Hai vector phải có cùng kích thước")
        
        # Tính tích vô hướng (dot product)
        dot_product = 0.0
        for i in range(len(vector1)):
            dot_product += vector1[i] * vector2[i]
        
        # Tính độ dài (norm) của mỗi vector
        norm1 = 0.0
        norm2 = 0.0
        for i in range(len(vector1)):
            norm1 += vector1[i] * vector1[i]
            norm2 += vector2[i] * vector2[i]
        
        norm1 = self._sqrt(norm1)
        norm2 = self._sqrt(norm2)
        
        # Tránh chia cho 0
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return similarity
    
    def manhattan_distance(self, vector1, vector2):

        if len(vector1) != len(vector2):
            raise ValueError("Hai vector phải có cùng kích thước")
        
        distance = 0.0
        for i in range(len(vector1)):
            diff = vector1[i] - vector2[i]
            # Tính giá trị tuyệt đối thủ công
            if diff < 0:
                diff = -diff
            distance += diff
        
        return distance
    
    def search_similar(self, query_vector, top_k=5, method='euclidean'):
        """
        Tìm kiếm các bài hát tương đồng với vector đầu vào.
        
        Args:
            query_vector (np.array): Vector đặc trưng của file cần tìm
            top_k (int): Số kết quả trả về
            method (str): Phương pháp tính khoảng cách ('euclidean', 'cosine', 'manhattan')
            
        Returns:
            list: Danh sách top_k bài hát tương đồng nhất
        """
        # Lấy tất cả vector từ database
        all_vectors = self.db.get_all_feature_vectors()
        
        if not all_vectors:
            return []
        
        # Tính khoảng cách/độ tương đồng với mỗi bài hát
        results = []
        
        for song_id, feature_vector in all_vectors:
            if method == 'euclidean':
                score = self.euclidean_distance(query_vector, feature_vector)
                # Khoảng cách nhỏ = tương đồng cao
                results.append((song_id, score, 'distance'))
            elif method == 'cosine':
                score = self.cosine_similarity(query_vector, feature_vector)
                # Similarity cao = tương đồng cao
                results.append((song_id, score, 'similarity'))
            elif method == 'manhattan':
                score = self.manhattan_distance(query_vector, feature_vector)
                results.append((song_id, score, 'distance'))
            else:
                raise ValueError(f"Phương pháp không hỗ trợ: {method}")
        
        # Sắp xếp kết quả
        if results[0][2] == 'distance':
            # Sắp xếp theo khoảng cách tăng dần (nhỏ nhất trước)
            results.sort(key=lambda x: x[1])
        else:
            # Sắp xếp theo độ tương đồng giảm dần (lớn nhất trước)
            results.sort(key=lambda x: x[1], reverse=True)
        
        # Lấy thông tin chi tiết của top_k kết quả
        top_results = []
        for i, (song_id, score, score_type) in enumerate(results[:top_k]):
            song_info = self.db.get_song_by_id(song_id)
            if song_info:
                song_info['score'] = score
                song_info['score_type'] = score_type
                song_info['rank'] = i + 1
                top_results.append(song_info)
        
        return top_results
    
    def search_by_audio_file(self, processed_data, top_k=5, method='euclidean'):

        query_vector = processed_data['feature_vector']
        return self.search_similar(query_vector, top_k, method)
    
    def classify_by_threshold(self, features, ste_threshold=0.01, zcr_threshold=0.1):

        ste_mean = features['ste_mean']
        zcr_mean = features['zcr_mean']
        
        # Xác định loại
        high_ste = ste_mean > ste_threshold
        high_zcr = zcr_mean > zcr_threshold
        
        if high_ste and not high_zcr:
            category = "Nhạc cụ / Âm nhạc"
            confidence = min(1.0, ste_mean / ste_threshold * 0.5 + (zcr_threshold - zcr_mean) / zcr_threshold * 0.5)
        elif not high_ste and high_zcr:
            category = "Tiếng nói"
            confidence = min(1.0, (ste_threshold - ste_mean) / ste_threshold * 0.5 + zcr_mean / zcr_threshold * 0.5)
        elif high_ste and high_zcr:
            category = "Âm thanh động / Nhiễu"
            confidence = min(1.0, (ste_mean / ste_threshold + zcr_mean / zcr_threshold) / 2)
        else:
            category = "Âm thanh tĩnh / Im lặng"
            confidence = min(1.0, ((ste_threshold - ste_mean) / ste_threshold + (zcr_threshold - zcr_mean) / zcr_threshold) / 2)
        
        return {
            'category': category,
            'confidence': max(0, min(1, confidence)),
            'ste_level': 'cao' if high_ste else 'thấp',
            'zcr_level': 'cao' if high_zcr else 'thấp',
            'ste_value': ste_mean,
            'zcr_value': zcr_mean
        }
    
    def find_duplicates(self, threshold=0.1):

        all_vectors = self.db.get_all_feature_vectors()
        duplicates = []
        
        for i in range(len(all_vectors)):
            for j in range(i + 1, len(all_vectors)):
                id1, vector1 = all_vectors[i]
                id2, vector2 = all_vectors[j]
                
                distance = self.euclidean_distance(vector1, vector2)
                
                if distance < threshold:
                    song1 = self.db.get_song_by_id(id1)
                    song2 = self.db.get_song_by_id(id2)
                    duplicates.append({
                        'song1': song1,
                        'song2': song2,
                        'distance': distance
                    })
        
        return duplicates
    
    def get_recommendations(self, song_id, top_k=5):

        song = self.db.get_song_by_id(song_id)
        if not song:
            return []
        
        query_vector = song['feature_vector']
        
        # Tìm kiếm và loại bỏ chính nó
        results = self.search_similar(query_vector, top_k + 1, method='euclidean')
        
        # Lọc bỏ bài hát gốc
        recommendations = [r for r in results if r['id'] != song_id][:top_k]
        
        return recommendations


# Test module
if __name__ == "__main__":
    print("=== Module Search Engine ===")
    
    # Test các hàm tính khoảng cách
    engine = SearchEngine()
    
    v1 = np.array([1.0, 2.0, 3.0, 4.0])
    v2 = np.array([2.0, 3.0, 4.0, 5.0])
    
    print(f"\nVector 1: {v1}")
    print(f"Vector 2: {v2}")
    print(f"Euclidean Distance: {engine.euclidean_distance(v1, v2):.4f}")
    print(f"Cosine Similarity: {engine.cosine_similarity(v1, v2):.4f}")
    print(f"Manhattan Distance: {engine.manhattan_distance(v1, v2):.4f}")
    
    print("\nCác hàm tìm kiếm có sẵn:")
    print("- search_similar(query_vector, top_k, method)")
    print("- search_by_audio_file(processed_data, top_k, method)")
    print("- classify_by_threshold(features, ste_threshold, zcr_threshold)")
    print("- find_duplicates(threshold)")
    print("- get_recommendations(song_id, top_k)")
