"""
Module 4: Giao diện người dùng & Trực quan hóa (UI/UX & Visualization)
Ứng dụng Desktop sử dụng PyQt5 với khả năng hiển thị biểu đồ STE/ZCR.
"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
    QTabWidget, QGroupBox, QLineEdit, QComboBox, QSlider, QSpinBox,
    QProgressBar, QStatusBar, QMessageBox, QSplitter, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QFormLayout,
    QDialogButtonBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import pygame

# Import các module khác
from audio_processing import load_audio, framing, calculate_ste_normalized, calculate_zcr, process_audio_file, extract_features, get_feature_vector
from database_manager import DatabaseManager
from search_engine import SearchEngine


class AudioProcessingThread(QThread):
    """Thread xử lý âm thanh để không block UI."""
    
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, file_path, frame_duration=25, overlap_ratio=0.5):
        super().__init__()
        self.file_path = file_path
        self.frame_duration = frame_duration
        self.overlap_ratio = overlap_ratio
    
    def run(self):
        try:
            self.progress.emit(20)
            result = process_audio_file(
                self.file_path, 
                self.frame_duration, 
                self.overlap_ratio
            )
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MplCanvas(FigureCanvas):
    """Canvas cho Matplotlib."""
    
    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)


class EditSongDialog(QDialog):
    """Dialog để chỉnh sửa thông tin bài hát."""
    
    def __init__(self, song_info, parent=None):
        super().__init__(parent)
        self.song_info = song_info
        self.setWindowTitle("Chỉnh sửa thông tin bài hát")
        self.setMinimumWidth(400)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QFormLayout(self)
        
        self.title_edit = QLineEdit(self.song_info.get('title', ''))
        self.artist_edit = QLineEdit(self.song_info.get('artist', ''))
        
        layout.addRow("Tên bài hát:", self.title_edit)
        layout.addRow("Ca sĩ:", self.artist_edit)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
    
    def get_data(self):
        return {
            'title': self.title_edit.text(),
            'artist': self.artist_edit.text()
        }


class MainWindow(QMainWindow):
    """Cửa sổ chính của ứng dụng."""
    
    def __init__(self):
        super().__init__()
        
        # Khoi tao pygame mixer de phat am thanh
        # Cau hinh cho file WAV: frequency=44100, size=-16, channels=2, buffer=2048
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        
        # Bien luu Sound object
        self.current_sound = None
        
        # Khoi tao database va search engine
        self.db = DatabaseManager()
        self.search_engine = SearchEngine(self.db)
        
        # Biến lưu trữ dữ liệu
        self.current_audio_data = None
        self.current_processed_data = None
        self.current_file_path = None
        
        # Setup UI
        self.setWindowTitle("Hệ thống Phân loại & Tìm kiếm Âm thanh - STE/ZCR")
        self.setMinimumSize(1200, 800)
        
        self.setup_ui()
        self.load_song_list()
        
        # Status bar
        self.statusBar().showMessage("Sẵn sàng")
    
    def setup_ui(self):
        """Thiết lập giao diện."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Splitter chia màn hình
        splitter = QSplitter(Qt.Horizontal)
        
        # Panel bên trái - Danh sách bài hát
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Panel bên phải - Phân tích & Visualization
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Thiết lập tỷ lệ
        splitter.setSizes([350, 850])
        
        main_layout.addWidget(splitter)
    
    def create_left_panel(self):
        """Tạo panel bên trái chứa danh sách bài hát."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tiêu đề
        title = QLabel("KHO NHẠC")
        title.setFont(QFont('Arial', 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Thanh tìm kiếm theo tên
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Tìm kiếm theo tên...")
        self.search_input.returnPressed.connect(self.search_by_name)
        search_btn = QPushButton("Tim")
        search_btn.clicked.connect(self.search_by_name)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(search_btn)
        layout.addLayout(search_layout)
        
        # Danh sách bài hát
        self.song_list = QListWidget()
        self.song_list.itemDoubleClicked.connect(self.on_song_double_click)
        self.song_list.itemSelectionChanged.connect(self.on_song_selected)
        layout.addWidget(self.song_list)
        
        # Các nút điều khiển
        btn_layout = QHBoxLayout()
        
        add_btn = QPushButton("Them")
        add_btn.clicked.connect(self.add_song_to_library)
        btn_layout.addWidget(add_btn)
        
        edit_btn = QPushButton("Sua")
        edit_btn.clicked.connect(self.edit_selected_song)
        btn_layout.addWidget(edit_btn)
        
        delete_btn = QPushButton("Xoa")
        delete_btn.clicked.connect(self.delete_selected_song)
        btn_layout.addWidget(delete_btn)
        
        layout.addLayout(btn_layout)
        
        # Thống kê
        self.stats_label = QLabel("Tổng: 0 bài hát")
        self.stats_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.stats_label)
        
        return panel
    
    def create_right_panel(self):
        """Tạo panel bên phải với tabs."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tabs
        self.tabs = QTabWidget()
        
        # Tab 1: Phân tích
        analysis_tab = self.create_analysis_tab()
        self.tabs.addTab(analysis_tab, "Phan tich")
        
        # Tab 2: Tìm kiếm
        search_tab = self.create_search_tab()
        self.tabs.addTab(search_tab, "Tim kiem tuong dong")
        
        # Tab 3: Thống kê
        stats_tab = self.create_stats_tab()
        self.tabs.addTab(stats_tab, "Thong ke")
        
        layout.addWidget(self.tabs)
        
        return panel
    
    def create_analysis_tab(self):
        """Tạo tab phân tích."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Khu vực upload
        upload_group = QGroupBox("Upload & Phan tich File")
        upload_layout = QHBoxLayout(upload_group)
        
        self.file_label = QLabel("Chưa chọn file")
        self.file_label.setStyleSheet("padding: 10px; border: 1px dashed #ccc; border-radius: 5px;")
        upload_layout.addWidget(self.file_label, 1)
        
        browse_btn = QPushButton("Chon File")
        browse_btn.clicked.connect(self.browse_file)
        upload_layout.addWidget(browse_btn)
        
        analyze_btn = QPushButton("Phan tich")
        analyze_btn.clicked.connect(self.analyze_current_file)
        upload_layout.addWidget(analyze_btn)
        
        layout.addWidget(upload_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Tham số phân tích
        params_group = QGroupBox("Tham so")
        params_layout = QHBoxLayout(params_group)
        
        params_layout.addWidget(QLabel("Frame (ms):"))
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setRange(10, 100)
        self.frame_spinbox.setValue(25)
        params_layout.addWidget(self.frame_spinbox)
        
        params_layout.addWidget(QLabel("Overlap:"))
        self.overlap_slider = QSlider(Qt.Horizontal)
        self.overlap_slider.setRange(0, 90)
        self.overlap_slider.setValue(50)
        params_layout.addWidget(self.overlap_slider)
        
        self.overlap_label = QLabel("50%")
        params_layout.addWidget(self.overlap_label)
        self.overlap_slider.valueChanged.connect(
            lambda v: self.overlap_label.setText(f"{v}%")
        )
        
        layout.addWidget(params_group)
        
        # Kết quả phân loại
        result_group = QGroupBox("Ket qua Phan loai")
        result_layout = QVBoxLayout(result_group)
        
        self.classification_label = QLabel("Chưa phân tích")
        self.classification_label.setFont(QFont('Arial', 12, QFont.Bold))
        self.classification_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.classification_label)
        
        self.features_label = QLabel("")
        self.features_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.features_label)
        
        layout.addWidget(result_group)
        
        # Biểu đồ
        charts_group = QGroupBox("Bieu do")
        charts_layout = QVBoxLayout(charts_group)
        
        # Canvas cho biểu đồ
        self.canvas = MplCanvas(self, width=10, height=6, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        charts_layout.addWidget(self.toolbar)
        charts_layout.addWidget(self.canvas)
        
        layout.addWidget(charts_group, 1)
        
        # Nút phát nhạc
        playback_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("Phat")
        self.play_btn.clicked.connect(self.play_audio)
        playback_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("Dung")
        self.stop_btn.clicked.connect(self.stop_audio)
        playback_layout.addWidget(self.stop_btn)
        
        self.save_btn = QPushButton("Luu vao Kho")
        self.save_btn.clicked.connect(self.save_to_library)
        playback_layout.addWidget(self.save_btn)
        
        layout.addLayout(playback_layout)
        
        return tab
    
    def create_search_tab(self):
        """Tạo tab tìm kiếm tương đồng."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Upload file để tìm kiếm
        search_group = QGroupBox("Tim kiem bai hat tuong tu")
        search_layout = QVBoxLayout(search_group)
        
        file_layout = QHBoxLayout()
        self.search_file_label = QLabel("Chọn file để tìm kiếm...")
        file_layout.addWidget(self.search_file_label, 1)
        
        browse_search_btn = QPushButton("Chon File")
        browse_search_btn.clicked.connect(self.browse_search_file)
        file_layout.addWidget(browse_search_btn)
        
        search_layout.addLayout(file_layout)
        
        # Tham số tìm kiếm
        params_layout = QHBoxLayout()
        
        params_layout.addWidget(QLabel("Phương pháp:"))
        self.search_method = QComboBox()
        self.search_method.addItems(["Euclidean", "Cosine", "Manhattan"])
        params_layout.addWidget(self.search_method)
        
        params_layout.addWidget(QLabel("Số kết quả:"))
        self.top_k_spinbox = QSpinBox()
        self.top_k_spinbox.setRange(1, 20)
        self.top_k_spinbox.setValue(5)
        params_layout.addWidget(self.top_k_spinbox)
        
        search_btn = QPushButton("Tim kiem")
        search_btn.clicked.connect(self.perform_search)
        params_layout.addWidget(search_btn)
        
        search_layout.addLayout(params_layout)
        layout.addWidget(search_group)
        
        # Kết quả tìm kiếm
        results_group = QGroupBox("Ket qua tim kiem")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Hạng", "Tên bài hát", "Ca sĩ", "Phân loại", "Điểm"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.itemDoubleClicked.connect(self.on_result_double_click)
        results_layout.addWidget(self.results_table)
        
        layout.addWidget(results_group, 1)
        
        return tab
    
    def create_stats_tab(self):
        """Tạo tab thống kê."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Thống kê tổng quan
        overview_group = QGroupBox("Tong quan")
        overview_layout = QVBoxLayout(overview_group)
        
        self.total_songs_label = QLabel("Tổng số bài hát: 0")
        self.total_songs_label.setFont(QFont('Arial', 12))
        overview_layout.addWidget(self.total_songs_label)
        
        self.total_duration_label = QLabel("Tổng thời lượng: 0 phút")
        self.total_duration_label.setFont(QFont('Arial', 12))
        overview_layout.addWidget(self.total_duration_label)
        
        layout.addWidget(overview_group)
        
        # Biểu đồ thống kê
        stats_chart_group = QGroupBox("Phan bo theo loai")
        stats_chart_layout = QVBoxLayout(stats_chart_group)
        
        self.stats_canvas = MplCanvas(self, width=8, height=4, dpi=100)
        stats_chart_layout.addWidget(self.stats_canvas)
        
        refresh_btn = QPushButton("Cap nhat thong ke")
        refresh_btn.clicked.connect(self.update_statistics)
        stats_chart_layout.addWidget(refresh_btn)
        
        layout.addWidget(stats_chart_group, 1)
        
        return tab
    
    def browse_file(self):
        """Mo dialog chon file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chon file am thanh", "",
            "Audio Files (*.wav *.mp3 *.ogg *.flac);;WAV Files (*.wav);;MP3 Files (*.mp3);;All Files (*)"
        )
        
        if file_path:
            self.current_file_path = file_path
            self.file_label.setText(os.path.basename(file_path))
    
    def analyze_current_file(self):
        """Phan tich file hien tai."""
        if not self.current_file_path:
            QMessageBox.warning(self, "Loi", "Vui long chon file truoc!")
            return
        
        # Kiem tra dinh dang file
        supported_formats = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac']
        file_ext = os.path.splitext(self.current_file_path)[1].lower()
        if file_ext not in supported_formats:
            QMessageBox.warning(self, "Loi", f"Dinh dang khong ho tro: {file_ext}\nChi ho tro: {', '.join(supported_formats)}")
            return
        
        # Hien thi progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Xu ly trong thread rieng
        frame_duration = self.frame_spinbox.value()
        overlap_ratio = self.overlap_slider.value() / 100.0
        
        self.process_thread = AudioProcessingThread(
            self.current_file_path,
            frame_duration,
            overlap_ratio
        )
        self.process_thread.progress.connect(self.progress_bar.setValue)
        self.process_thread.finished.connect(self.on_analysis_complete)
        self.process_thread.error.connect(self.on_analysis_error)
        self.process_thread.start()
    
    def on_analysis_complete(self, result):
        """Xử lý khi phân tích hoàn tất."""
        self.progress_bar.setVisible(False)
        self.current_processed_data = result
        self.current_audio_data = result['audio_data']
        
        # Hiển thị kết quả phân loại
        classification = result['classification']
        self.classification_label.setText(f"{classification}")
        
        # Hiển thị đặc trưng
        features = result['features']
        features_text = (
            f"STE: Mean={features['ste_mean']:.6f}, Std={features['ste_std']:.6f}\n"
            f"ZCR: Mean={features['zcr_mean']:.4f}, Std={features['zcr_std']:.4f}\n"
            f"Thời lượng: {features['duration']:.2f}s | Frames: {features['num_frames']}"
        )
        self.features_label.setText(features_text)
        
        # Vẽ biểu đồ
        self.plot_analysis(result)
        
        self.statusBar().showMessage("Phân tích hoàn tất!")
    
    def on_analysis_error(self, error_msg):
        """Xử lý lỗi khi phân tích."""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Lỗi", f"Không thể phân tích file:\n{error_msg}")
    
    def plot_analysis(self, result):
        """Vẽ biểu đồ phân tích."""
        self.canvas.fig.clear()
        
        audio_data = result['audio_data']
        sample_rate = result['sample_rate']
        features = result['features']
        
        # Tạo 3 subplot
        ax1 = self.canvas.fig.add_subplot(3, 1, 1)
        ax2 = self.canvas.fig.add_subplot(3, 1, 2)
        ax3 = self.canvas.fig.add_subplot(3, 1, 3)
        
        # Plot 1: Waveform
        time_axis = np.arange(len(audio_data)) / sample_rate
        ax1.plot(time_axis, audio_data, color='steelblue', linewidth=0.5)
        ax1.set_title('Dạng sóng âm thanh (Waveform)', fontsize=10)
        ax1.set_xlabel('Thời gian (s)')
        ax1.set_ylabel('Biên độ')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: STE
        ste_values = features['ste']
        frame_times = np.linspace(0, features['duration'], len(ste_values))
        ax2.plot(frame_times, ste_values, color='orangered', linewidth=1)
        ax2.fill_between(frame_times, ste_values, alpha=0.3, color='orangered')
        ax2.set_title('Năng lượng ngắn hạn (STE)', fontsize=10)
        ax2.set_xlabel('Thời gian (s)')
        ax2.set_ylabel('STE')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: ZCR
        zcr_values = features['zcr']
        ax3.plot(frame_times, zcr_values, color='forestgreen', linewidth=1)
        ax3.fill_between(frame_times, zcr_values, alpha=0.3, color='forestgreen')
        ax3.set_title('Tốc độ qua điểm không (ZCR)', fontsize=10)
        ax3.set_xlabel('Thời gian (s)')
        ax3.set_ylabel('ZCR')
        ax3.grid(True, alpha=0.3)
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()
    
    def play_audio(self):
        """Phat am thanh."""
        if self.current_file_path and os.path.exists(self.current_file_path):
            try:
                # Dung nhac dang phat truoc
                self.stop_audio()
                
                # Kiem tra dinh dang file
                if self.current_file_path.lower().endswith('.wav'):
                    # Su dung Sound object cho file WAV (chinh xac hon)
                    self.current_sound = pygame.mixer.Sound(self.current_file_path)
                    self.current_sound.play()
                else:
                    # Su dung music cho cac dinh dang khac
                    pygame.mixer.music.load(self.current_file_path)
                    pygame.mixer.music.play()
                
                self.statusBar().showMessage("Dang phat...")
            except Exception as e:
                QMessageBox.warning(self, "Loi", f"Khong the phat: {e}")
    
    def stop_audio(self):
        """Dung phat am thanh."""
        # Dung Sound object neu dang phat
        if self.current_sound:
            self.current_sound.stop()
            self.current_sound = None
        # Dung music
        pygame.mixer.music.stop()
        self.statusBar().showMessage("Da dung")
    
    def save_to_library(self):
        """Lưu bài hát vào kho."""
        if not self.current_processed_data:
            QMessageBox.warning(self, "Lỗi", "Vui lòng phân tích file trước!")
            return
        
        # Mở dialog nhập thông tin
        dialog = EditSongDialog({
            'title': os.path.splitext(os.path.basename(self.current_file_path))[0],
            'artist': ''
        }, self)
        
        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_data()
            song_id = self.db.add_song(
                self.current_file_path,
                self.current_processed_data,
                title=data['title'],
                artist=data['artist']
            )
            
            if song_id:
                QMessageBox.information(self, "Thành công", "Đã lưu vào kho nhạc!")
                self.load_song_list()
            else:
                QMessageBox.warning(self, "Lỗi", "Không thể lưu bài hát!")
    
    def load_song_list(self):
        """Load danh sách bài hát từ database."""
        self.song_list.clear()
        songs = self.db.get_all_songs()
        
        for song in songs:
            item = QListWidgetItem(
                f"{song['title']}" + (f" - {song['artist']}" if song['artist'] else "")
            )
            item.setData(Qt.UserRole, song['id'])
            self.song_list.addItem(item)
        
        # Cập nhật thống kê
        stats = self.db.get_statistics()
        self.stats_label.setText(f"Tổng: {stats['total_songs']} bài hát")
    
    def search_by_name(self):
        """Tìm kiếm bài hát theo tên."""
        keyword = self.search_input.text().strip()
        
        if not keyword:
            self.load_song_list()
            return
        
        self.song_list.clear()
        songs = self.db.search_by_name(keyword)
        
        for song in songs:
            item = QListWidgetItem(
                f"{song['title']}" + (f" - {song['artist']}" if song['artist'] else "")
            )
            
            item.setData(Qt.UserRole, song['id'])
            self.song_list.addItem(item)
    
    def on_song_selected(self):
        """Xử lý khi chọn bài hát."""
        pass
    
    def on_song_double_click(self, item):
        """Xử lý double click vào bài hát."""
        song_id = item.data(Qt.UserRole)
        song = self.db.get_song_by_id(song_id)
        
        if song and os.path.exists(song['file_path']):
            self.current_file_path = song['file_path']
            self.file_label.setText(song['file_name'])
            
            # Hiển thị thông tin đã lưu
            self.classification_label.setText(f"{song['classification']}")
            
            features_text = (
                f"STE: Mean={song['ste_mean']:.6f}, Std={song['ste_std']:.6f}\n"
                f"ZCR: Mean={song['zcr_mean']:.4f}, Std={song['zcr_std']:.4f}\n"
                f"Thời lượng: {song['duration']:.2f}s"
            )
            self.features_label.setText(features_text)
            
            # Vẽ biểu đồ từ dữ liệu đã lưu
            self.plot_saved_analysis(song)
            
            # Phát nhạc
            self.play_audio()
    
    def plot_saved_analysis(self, song):
        """Vẽ biểu đồ từ dữ liệu đã lưu."""
        self.canvas.fig.clear()
        
        # Tạo 2 subplot (không có waveform vì không lưu)
        ax1 = self.canvas.fig.add_subplot(2, 1, 1)
        ax2 = self.canvas.fig.add_subplot(2, 1, 2)
        
        # Plot STE
        ste_values = song['ste_data']
        frame_times = np.linspace(0, song['duration'], len(ste_values))
        ax1.plot(frame_times, ste_values, color='orangered', linewidth=1)
        ax1.fill_between(frame_times, ste_values, alpha=0.3, color='orangered')
        ax1.set_title('Năng lượng ngắn hạn (STE)', fontsize=10)
        ax1.set_xlabel('Thời gian (s)')
        ax1.set_ylabel('STE')
        ax1.grid(True, alpha=0.3)
        
        # Plot ZCR
        zcr_values = song['zcr_data']
        ax2.plot(frame_times, zcr_values, color='forestgreen', linewidth=1)
        ax2.fill_between(frame_times, zcr_values, alpha=0.3, color='forestgreen')
        ax2.set_title('Tốc độ qua điểm không (ZCR)', fontsize=10)
        ax2.set_xlabel('Thời gian (s)')
        ax2.set_ylabel('ZCR')
        ax2.grid(True, alpha=0.3)
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()
    
    def add_song_to_library(self):
        """Them bai hat moi vao kho."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Chon file am thanh", "",
            "Audio Files (*.wav *.mp3 *.ogg *.flac);;WAV Files (*.wav);;MP3 Files (*.mp3);;All Files (*)"
        )
        
        if not file_paths:
            return
        
        supported_formats = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac']
        added = 0
        for file_path in file_paths:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in supported_formats:
                try:
                    processed = process_audio_file(file_path)
                    song_id = self.db.add_song(file_path, processed)
                    if song_id:
                        added += 1
                except Exception as e:
                    print(f"Loi xu ly {file_path}: {e}")
        
        if added > 0:
            QMessageBox.information(self, "Thanh cong", f"Da them {added} bai hat!")
            self.load_song_list()
    
    def edit_selected_song(self):
        """Chỉnh sửa bài hát được chọn."""
        current_item = self.song_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn bài hát!")
            return
        
        song_id = current_item.data(Qt.UserRole)
        song = self.db.get_song_by_id(song_id)
        
        if song:
            dialog = EditSongDialog(song, self)
            if dialog.exec_() == QDialog.Accepted:
                data = dialog.get_data()
                self.db.update_song(song_id, title=data['title'], artist=data['artist'])
                self.load_song_list()
    
    def delete_selected_song(self):
        """Xóa bài hát được chọn."""
        current_item = self.song_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn bài hát!")
            return
        
        reply = QMessageBox.question(
            self, "Xác nhận",
            "Bạn có chắc muốn xóa bài hát này?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            song_id = current_item.data(Qt.UserRole)
            if self.db.delete_song(song_id):
                self.load_song_list()
    
    def browse_search_file(self):
        """Chon file de tim kiem."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chon file am thanh", "",
            "Audio Files (*.wav *.mp3 *.ogg *.flac);;WAV Files (*.wav);;MP3 Files (*.mp3);;All Files (*)"
        )
        
        if file_path:
            self.search_file_path = file_path
            self.search_file_label.setText(os.path.basename(file_path))
    
    def perform_search(self):
        """Thực hiện tìm kiếm."""
        if not hasattr(self, 'search_file_path') or not self.search_file_path:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn file!")
            return
        
        try:
            # Xử lý file query
            processed = process_audio_file(self.search_file_path)
            
            # Lấy phương pháp và top_k
            method = self.search_method.currentText().lower()
            top_k = self.top_k_spinbox.value()
            
            # Tìm kiếm
            results = self.search_engine.search_by_audio_file(
                processed, top_k, method
            )
            
            # Hiển thị kết quả
            self.results_table.setRowCount(len(results))
            
            for i, result in enumerate(results):
                self.results_table.setItem(i, 0, QTableWidgetItem(str(result['rank'])))
                self.results_table.setItem(i, 1, QTableWidgetItem(result['title']))
                self.results_table.setItem(i, 2, QTableWidgetItem(result['artist'] or ''))
                self.results_table.setItem(i, 3, QTableWidgetItem(result['classification']))
                self.results_table.setItem(i, 4, QTableWidgetItem(f"{result['score']:.4f}"))
            
            self.statusBar().showMessage(f"Tìm thấy {len(results)} kết quả")
            
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể tìm kiếm:\n{e}")
    
    def on_result_double_click(self, item):
        """Xu ly double click vao ket qua tim kiem."""
        row = item.row()
        title = self.results_table.item(row, 1).text()
        
        # Tim va phat bai hat
        songs = self.db.search_by_name(title)
        if songs:
            song = songs[0]
            if os.path.exists(song['file_path']):
                self.current_file_path = song['file_path']
                self.play_audio()
    
    def update_statistics(self):
        """Cập nhật thống kê."""
        stats = self.db.get_statistics()
        
        self.total_songs_label.setText(f"Tổng số bài hát: {stats['total_songs']}")
        
        total_minutes = stats['total_duration'] / 60
        self.total_duration_label.setText(f"Tổng thời lượng: {total_minutes:.1f} phút")
        
        # Vẽ biểu đồ tròn
        self.stats_canvas.fig.clear()
        ax = self.stats_canvas.fig.add_subplot(1, 1, 1)
        
        by_class = stats['by_classification']
        if by_class:
            labels = list(by_class.keys())
            sizes = list(by_class.values())
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
            
            ax.pie(sizes, labels=labels, colors=colors[:len(labels)],
                   autopct='%1.1f%%', startangle=90)
            ax.set_title('Phân bố theo loại âm thanh')
        else:
            ax.text(0.5, 0.5, 'Chưa có dữ liệu', ha='center', va='center')
        
        self.stats_canvas.draw()
    
    def closeEvent(self, event):
        """Xử lý khi đóng ứng dụng."""
        pygame.mixer.quit()
        self.db.close()
        event.accept()


def main():
    """Hàm chính chạy ứng dụng."""
    app = QApplication(sys.argv)
    
    # Set style
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
