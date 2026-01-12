"""
Module 2: Quản lý dữ liệu & Kho nhạc (Data Management)
Lưu trữ và quản lý thông tin các bài hát đã được phân tích.
Sử dụng SQLite làm database.
"""

import sqlite3
import json
import os
import numpy as np
from datetime import datetime


class DatabaseManager:

    def __init__(self, db_path="audio_database.db"):
     
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Tạo kết nối đến database."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
    
    def _create_tables(self):
        # Bảng lưu thông tin bài hát
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS songs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_name TEXT NOT NULL,
                title TEXT,
                artist TEXT,
                duration REAL,
                sample_rate INTEGER,
                classification TEXT,
                feature_vector TEXT,
                ste_data TEXT,
                zcr_data TEXT,
                ste_mean REAL,
                ste_std REAL,
                ste_max REAL,
                ste_min REAL,
                zcr_mean REAL,
                zcr_std REAL,
                zcr_max REAL,
                zcr_min REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Bảng lưu lịch sử tìm kiếm
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_file TEXT,
                results TEXT,
                search_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def add_song(self, file_path, processed_data, title=None, artist=None):

        try:
            features = processed_data['features']
            feature_vector = processed_data['feature_vector']
            
            file_name = os.path.basename(file_path)
            if title is None:
                title = os.path.splitext(file_name)[0]
            
            # Chuyển đổi các mảng numpy thành JSON để lưu trữ
            feature_vector_json = json.dumps(feature_vector.tolist())
            ste_data_json = json.dumps(features['ste'].tolist())
            zcr_data_json = json.dumps(features['zcr'].tolist())
            
            self.cursor.execute('''
                INSERT OR REPLACE INTO songs (
                    file_path, file_name, title, artist, duration, sample_rate,
                    classification, feature_vector, ste_data, zcr_data,
                    ste_mean, ste_std, ste_max, ste_min,
                    zcr_mean, zcr_std, zcr_max, zcr_min, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                file_path, file_name, title, artist,
                features['duration'], processed_data['sample_rate'],
                processed_data['classification'],
                feature_vector_json, ste_data_json, zcr_data_json,
                features['ste_mean'], features['ste_std'],
                features['ste_max'], features['ste_min'],
                features['zcr_mean'], features['zcr_std'],
                features['zcr_max'], features['zcr_min'],
                datetime.now()
            ))
            
            self.conn.commit()
            return self.cursor.lastrowid
            
        except Exception as e:
            print(f"Lỗi khi thêm bài hát: {e}")
            return None
    
    def get_song_by_id(self, song_id):

        self.cursor.execute('SELECT * FROM songs WHERE id = ?', (song_id,))
        row = self.cursor.fetchone()
        
        if row:
            return self._row_to_dict(row)
        return None
    
    def get_song_by_path(self, file_path):
        """
        Lấy thông tin bài hát theo đường dẫn file.
        
        Args:
            file_path (str): Đường dẫn file
            
        Returns:
            dict: Thông tin bài hát
        """
        self.cursor.execute('SELECT * FROM songs WHERE file_path = ?', (file_path,))
        row = self.cursor.fetchone()
        
        if row:
            return self._row_to_dict(row)
        return None
    
    def get_all_songs(self):

        self.cursor.execute('SELECT * FROM songs ORDER BY title')
        rows = self.cursor.fetchall()
        
        return [self._row_to_dict(row) for row in rows]
    
    def update_song(self, song_id, title=None, artist=None):

        updates = []
        values = []
        
        if title is not None:
            updates.append('title = ?')
            values.append(title)
        
        if artist is not None:
            updates.append('artist = ?')
            values.append(artist)
        
        if not updates:
            return False
        
        updates.append('updated_at = ?')
        values.append(datetime.now())
        values.append(song_id)
        
        query = f"UPDATE songs SET {', '.join(updates)} WHERE id = ?"
        
        try:
            self.cursor.execute(query, values)
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Lỗi khi cập nhật: {e}")
            return False
    
    def delete_song(self, song_id):

        try:
            self.cursor.execute('DELETE FROM songs WHERE id = ?', (song_id,))
            self.conn.commit()
            return self.cursor.rowcount > 0
        except Exception as e:
            print(f"Lỗi khi xóa: {e}")
            return False
    
    def search_by_name(self, keyword):

        self.cursor.execute('''
            SELECT * FROM songs 
            WHERE title LIKE ? OR artist LIKE ? OR file_name LIKE ?
            ORDER BY title
        ''', (f'%{keyword}%', f'%{keyword}%', f'%{keyword}%'))
        
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def get_songs_by_classification(self, classification):

        self.cursor.execute(
            'SELECT * FROM songs WHERE classification = ? ORDER BY title',
            (classification,)
        )
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def get_all_feature_vectors(self):

        self.cursor.execute('SELECT id, feature_vector FROM songs')
        rows = self.cursor.fetchall()
        
        result = []
        for row in rows:
            song_id = row[0]
            feature_vector = np.array(json.loads(row[1]))
            result.append((song_id, feature_vector))
        
        return result
    
    def save_search_history(self, query_file, results):

        try:
            results_json = json.dumps(results)
            self.cursor.execute(
                'INSERT INTO search_history (query_file, results) VALUES (?, ?)',
                (query_file, results_json)
            )
            self.conn.commit()
        except Exception as e:
            print(f"Lỗi lưu lịch sử: {e}")
    
    def get_statistics(self):
      
        stats = {}
        
        # Tổng số bài hát
        self.cursor.execute('SELECT COUNT(*) FROM songs')
        stats['total_songs'] = self.cursor.fetchone()[0]
        
        # Số bài hát theo phân loại
        self.cursor.execute('''
            SELECT classification, COUNT(*) 
            FROM songs 
            GROUP BY classification
        ''')
        stats['by_classification'] = dict(self.cursor.fetchall())
        
        # Tổng thời lượng
        self.cursor.execute('SELECT SUM(duration) FROM songs')
        total_duration = self.cursor.fetchone()[0]
        stats['total_duration'] = total_duration if total_duration else 0
        
        return stats
    
    def _row_to_dict(self, row):

        columns = [
            'id', 'file_path', 'file_name', 'title', 'artist', 'duration',
            'sample_rate', 'classification', 'feature_vector', 'ste_data',
            'zcr_data', 'ste_mean', 'ste_std', 'ste_max', 'ste_min',
            'zcr_mean', 'zcr_std', 'zcr_max', 'zcr_min', 'created_at', 'updated_at'
        ]
        
        song_dict = dict(zip(columns, row))
        
        # Parse JSON fields
        if song_dict['feature_vector']:
            song_dict['feature_vector'] = np.array(json.loads(song_dict['feature_vector']))
        if song_dict['ste_data']:
            song_dict['ste_data'] = np.array(json.loads(song_dict['ste_data']))
        if song_dict['zcr_data']:
            song_dict['zcr_data'] = np.array(json.loads(song_dict['zcr_data']))
        
        return song_dict
    
    def close(self):
        """Đóng kết nối database."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Test module
if __name__ == "__main__":
    print("=== Module Quản lý Database ===")
    
    # Tạo database test
    db = DatabaseManager("test_database.db")
    
    print(f"Database path: {db.db_path}")
    print(f"Statistics: {db.get_statistics()}")
    
    db.close()
    print("Database đã được tạo thành công!")
