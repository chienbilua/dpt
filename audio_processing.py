"""
Module 1: Xu ly tin hieu co so (Core Processing)
Chua cac thuat toan STE, ZCR va cac ham xu ly am thanh co ban.
"""

import numpy as np
from scipy.io import wavfile
import os

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


def load_audio(file_path):

    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.wav':
        sample_rate, audio_data = wavfile.read(file_path)
        
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float64) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float64) / 2147483648.0
        elif audio_data.dtype == np.uint8:
            audio_data = (audio_data.astype(np.float64) - 128) / 128.0
            
    elif file_ext in ['.mp3', '.ogg', '.flac', '.m4a', '.aac']:
        if not PYDUB_AVAILABLE:
            raise ImportError("Can cai dat pydub de doc file MP3: pip install pydub")
        
        audio = AudioSegment.from_file(file_path)
        
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        sample_rate = audio.frame_rate
        
        samples = np.array(audio.get_array_of_samples())
        
        # Chuan hoa ve [-1, 1] dua tren sample width
        if audio.sample_width == 1:  # 8-bit
            audio_data = (samples.astype(np.float64) - 128) / 128.0
        elif audio.sample_width == 2:  # 16-bit
            audio_data = samples.astype(np.float64) / 32768.0
        elif audio.sample_width == 4:  # 32-bit
            audio_data = samples.astype(np.float64) / 2147483648.0
        else:
            audio_data = samples.astype(np.float64) / np.max(np.abs(samples))
    else:
        raise ValueError(f"Dinh dang file khong ho tro: {file_ext}")
    
    return sample_rate, audio_data


def framing(audio_data, sample_rate, frame_duration_ms=25, overlap_ratio=0.5):

    
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    
    hop_size = int(frame_size * (1 - overlap_ratio))
    
    hop_size = max(1, hop_size)
    
    # Tính số khung
    num_frames = 1 + (len(audio_data) - frame_size) // hop_size
    
    if num_frames <= 0:
        frame = np.zeros(frame_size)
        frame[:len(audio_data)] = audio_data
        return np.array([frame])
    
    frames = np.zeros((num_frames, frame_size))
    
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frames[i] = audio_data[start:end]
    
    return frames


def calculate_ste(frames):

    num_frames = frames.shape[0]
    ste_values = np.zeros(num_frames)
    
    for i in range(num_frames):
        frame = frames[i]
        # Tính tổng bình phương các mẫu trong khung
        energy = 0.0
        for sample in frame:
            energy += sample * sample
        ste_values[i] = energy
    
    return ste_values


def calculate_ste_normalized(frames):

    num_frames = frames.shape[0]
    frame_size = frames.shape[1]
    ste_values = np.zeros(num_frames)
    
    for i in range(num_frames):
        frame = frames[i]
        energy = 0.0
        for sample in frame:
            energy += sample * sample
        ste_values[i] = energy / frame_size
    
    return ste_values


def calculate_zcr(frames):

    num_frames = frames.shape[0]
    zcr_values = np.zeros(num_frames)
    
    for i in range(num_frames):
        frame = frames[i]
        frame_size = len(frame)
        zero_crossings = 0
        
        for j in range(1, frame_size):
            # Đếm số lần tín hiệu đổi dấu
            if (frame[j] >= 0 and frame[j-1] < 0) or (frame[j] < 0 and frame[j-1] >= 0):
                zero_crossings += 1
        
        # Chuẩn hóa bằng độ dài khung
        zcr_values[i] = zero_crossings / (frame_size - 1)
    
    return zcr_values


def sign_function(x):

    if x >= 0:
        return 1
    else:
        return -1


def extract_features(audio_data, sample_rate, frame_duration_ms=25, overlap_ratio=0.5):

    # Phân khung
    frames = framing(audio_data, sample_rate, frame_duration_ms, overlap_ratio)
    
    # Tính STE và ZCR
    ste_values = calculate_ste_normalized(frames)
    zcr_values = calculate_zcr(frames)
    
    # Tạo vector đặc trưng tổng hợp
    features = {
        'ste': ste_values,
        'zcr': zcr_values,
        'ste_mean': float(np.mean(ste_values)),
        'ste_std': float(np.std(ste_values)),
        'ste_max': float(np.max(ste_values)),
        'ste_min': float(np.min(ste_values)),
        'zcr_mean': float(np.mean(zcr_values)),
        'zcr_std': float(np.std(zcr_values)),
        'zcr_max': float(np.max(zcr_values)),
        'zcr_min': float(np.min(zcr_values)),
        'num_frames': len(frames),
        'duration': len(audio_data) / sample_rate
    }
    
    return features


def get_feature_vector(features):

    return np.array([
        features['ste_mean'],
        features['ste_std'],
        features['ste_max'],
        features['ste_min'],
        features['zcr_mean'],
        features['zcr_std'],
        features['zcr_max'],
        features['zcr_min']
    ])


def classify_audio(features, ste_threshold=0.01, zcr_threshold=0.1):

    ste_mean = features['ste_mean']
    zcr_mean = features['zcr_mean']
    
    if ste_mean > ste_threshold:
        if zcr_mean > zcr_threshold:
            return "Âm thanh động / Nhiễu"
        else:
            return "Nhạc cụ / Âm nhạc"
    else:
        if zcr_mean > zcr_threshold:
            return "Tiếng nói"
        else:
            return "Âm thanh tĩnh / Im lặng"


def process_audio_file(file_path, frame_duration_ms=25, overlap_ratio=0.5):

    # Đọc file
    sample_rate, audio_data = load_audio(file_path)
    
    # Trích xuất đặc trưng
    features = extract_features(audio_data, sample_rate, frame_duration_ms, overlap_ratio)
    
    # Phân loại
    classification = classify_audio(features)
    
    return {
        'file_path': file_path,
        'sample_rate': sample_rate,
        'audio_data': audio_data,
        'features': features,
        'feature_vector': get_feature_vector(features),
        'classification': classification
    }


# Test module nếu chạy trực tiếp
if __name__ == "__main__":
    print("=== Module Xử lý Tín hiệu Âm thanh ===")
    print("Các hàm có sẵn:")
    print("- load_audio(file_path): Đọc file .wav")
    print("- framing(audio_data, sample_rate): Chia thành các khung")
    print("- calculate_ste(frames): Tính năng lượng ngắn hạn")
    print("- calculate_zcr(frames): Tính tốc độ qua điểm không")
    print("- extract_features(audio_data, sample_rate): Trích xuất đặc trưng")
    print("- classify_audio(features): Phân loại âm thanh")
    print("- process_audio_file(file_path): Xử lý hoàn chỉnh file")
