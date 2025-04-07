import cv2
import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import json
import os
from ultralytics import YOLO
import numpy as np
import time
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import threading
import base64
from functools import wraps

# 在PackageScannerApp类之前添加Flask应用
app = Flask(__name__)
CORS(app)

# 动态下载模型（放在全局作用域，避免每次请求重复下载）
MODEL_URL = 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt'
model = YOLO(MODEL_URL)  # 首次运行时自动下载
# 添加用户验证相关代码
users = {
    'admin': {'password': 'admin123', 'role': 'admin'},
    'user': {'password': 'user123', 'role': 'user'}
}


class PackageScannerApp:
    def __init__(self, window):
        self.window = window
        self.window.title("智慧物流识别系统")

        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        # 设置摄像头分辨率和帧率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 初始化YOLO模型
        self.model = YOLO('yolov8n.pt')  # 加载预训练模型
        # 设置推理大小以提升速度
        self.model.predict(None, imgsz=320)

        # 初始化二维码检测器
        self.qr_detector = cv2.QRCodeDetector()

        # 初始化FPS计算
        self.fps = 0
        self.fps_time = time.time()
        self.frame_count = 0
        # 添加处理计数器
        self.process_count = 0
        self.save_count = 0

        # 创建GUI组件
        self.create_widgets()

        # 初始化记录存储
        self.scan_history = []
        self.load_history()

        # 开始视频流
        self.update_video()

    def create_widgets(self):
        # 创建左侧视频显示区域
        self.video_frame = ttk.Frame(self.window)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10)

        self.video_label = ttk.Label(self.video_frame)
        self.video_label.grid(row=0, column=0)

        # FPS显示
        self.fps_label = ttk.Label(self.video_frame, text="FPS: 0")
        self.fps_label.grid(row=1, column=0, pady=5)

        # 创建右侧信息显示区域
        self.info_frame = ttk.Frame(self.window)
        self.info_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        # 当前扫描结果
        ttk.Label(self.info_frame, text="当前扫描结果").grid(row=0, column=0, pady=5)
        self.current_result = tk.Text(self.info_frame, height=5, width=40)
        self.current_result.grid(row=1, column=0, pady=5)

        # 物体识别结果
        ttk.Label(self.info_frame, text="物体识别结果").grid(row=2, column=0, pady=5)
        self.object_result = tk.Text(self.info_frame, height=5, width=40)
        self.object_result.grid(row=3, column=0, pady=5)

        # 历史记录
        ttk.Label(self.info_frame, text="扫描历史").grid(row=4, column=0, pady=5)
        self.history_text = tk.Text(self.info_frame, height=15, width=40)
        self.history_text.grid(row=5, column=0, pady=5)

        # 控制按钮
        self.control_frame = ttk.Frame(self.info_frame)
        self.control_frame.grid(row=6, column=0, pady=10)

        ttk.Button(self.control_frame, text="清除历史",
                   command=self.clear_history).grid(row=0, column=0, padx=5)
        ttk.Button(self.control_frame, text="保存记录",
                   command=self.save_history).grid(row=0, column=1, padx=5)

    def detect_objects(self, frame):
        # 每30帧进行一次物体检测
        if self.process_count % 3 != 0:
            return frame, []

        # 使用YOLO进行物体检测
        results = self.model(frame)

        detected_objects = []

        # 处理检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # 获取置信度
                conf = box.conf[0].cpu().numpy()

                # 获取类别
                cls = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names[cls]

                if conf > 0.6:  # 提高置信度阈值，减少误识别
                    # 在图像上绘制边界框
                    cv2.rectangle(frame,
                                  (int(x1), int(y1)),
                                  (int(x2), int(y2)),
                                  (255, 0, 0), 2)

                    # 添加标签
                    label = f"{cls_name}: {conf:.2f}"
                    cv2.putText(frame, label,
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 0), 2)

                    detected_objects.append({
                        "class": cls_name,
                        "confidence": float(conf)
                    })

        return frame, detected_objects

    def detect_qr_codes(self, frame):
        # 每30帧进行一次二维码识别
        if self.process_count % 30 != 0:
            return []

        # 使用OpenCV检测和解码二维码
        decoded_info, points, _ = self.qr_detector.detectAndDecode(frame)

        barcodes = []
        if points is not None and len(decoded_info) > 0:
            points = points.astype(int)
            # 绘制二维码边框
            cv2.polylines(frame, [points], True, (0, 255, 0), 2)

            # 显示文本
            text = f"QR: {decoded_info}"
            cv2.putText(frame, text, (points[0][0][0], points[0][0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            barcodes.append({
                "type": "QRCODE",
                "data": decoded_info,
                "points": points.tolist()
            })

        return barcodes

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # 增加处理计数
            self.process_count += 1

            # 计算FPS
            self.frame_count += 1
            if time.time() - self.fps_time > 1:
                self.fps = self.frame_count
                self.frame_count = 0
                self.fps_time = time.time()
                self.fps_label.config(text=f"FPS: {self.fps}")

            # 物体检测
            frame, detected_objects = self.detect_objects(frame)

            # 更新物体识别结果显示
            # 每30帧更新一次识别结果
            if self.process_count % 30 == 0:
                self.update_object_result(detected_objects)

            # 二维码识别
            barcodes = self.detect_qr_codes(frame)

            # 处理找到的二维码
            for barcode in barcodes:
                # 更新当前结果显示
                self.update_current_result(barcode["type"], barcode["data"])

            # 在画面上显示FPS
            cv2.putText(frame, f"FPS: {self.fps}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 转换图像格式用于显示
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # 提高更新频率
        self.window.after(30, self.update_video)  # 降低到30ms

    def update_object_result(self, detected_objects):
        self.object_result.delete(1.0, tk.END)
        if detected_objects:
            for obj in detected_objects:
                self.object_result.insert(tk.END,
                                          f"物体: {obj['class']}\n"
                                          f"置信度: {obj['confidence']:.2f}\n"
                                          f"------------------------\n"
                                          )

    def update_current_result(self, barcode_type, barcode_data):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 检查是否与上一次扫描结果相同
        if self.scan_history and \
                self.scan_history[-1]['type'] == barcode_type and \
                self.scan_history[-1]['data'] == barcode_data:
            return  # 如果相同则不更新

        result = f"类型: {barcode_type}\n数据: {barcode_data}\n时间: {timestamp}\n"

        self.current_result.delete(1.0, tk.END)
        self.current_result.insert(tk.END, result)

        # 添加到历史记录
        self.scan_history.append({
            "type": barcode_type,
            "data": barcode_data,
            "timestamp": timestamp
        })
        # 每次扫描都更新显示
        self.update_history_display()

    def update_history_display(self):
        self.history_text.delete(1.0, tk.END)
        # 显示最近15条不重复的记录
        shown_items = []
        shown_count = 0

        for item in reversed(self.scan_history):
            # 检查是否已经显示过相同的记录
            item_key = f"{item['type']}_{item['data']}"
            if item_key not in shown_items:
                self.history_text.insert(tk.END,
                                         f"类型: {item['type']}\n"
                                         f"数据: {item['data']}\n"
                                         f"时间: {item['timestamp']}\n"
                                         f"------------------------\n"
                                         )
                shown_items.append(item_key)
                shown_count += 1

            if shown_count >= 15:  # 限制显示数量
                break

    def clear_history(self):
        self.scan_history = []
        self.update_history_display()

    def save_history(self):
        # 添加延迟以避免频繁保存
        if not hasattr(self, 'last_save_time') or \
                time.time() - self.last_save_time > 1:  # 1秒内不重复保存
            self.last_save_time = time.time()
            # 只保存最近的100条不重复记录
            unique_records = []
            seen_records = set()

            for item in reversed(self.scan_history):
                item_key = f"{item['type']}_{item['data']}"
                if item_key not in seen_records:
                    unique_records.append(item)
                    seen_records.add(item_key)

                    if len(unique_records) >= 100:
                        break

            with open("scan_history.json", "w", encoding="utf-8") as f:
                json.dump(unique_records, f, ensure_ascii=False, indent=2)

    def load_history(self):
        try:
            if os.path.exists("scan_history.json"):
                with open("scan_history.json", "r", encoding="utf-8") as f:
                    self.scan_history = json.load(f)
                self.update_history_display()
        except Exception as e:
            print(f"加载历史记录失败: {e}")

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

    def get_frame_data(self):
        try:
            ret, frame = self.cap.read()
            if ret:
                # 进行物体检测
                frame, detected_objects = self.detect_objects(frame)

                # 二维码识别
                decoded_info, points, _ = self.qr_detector.detectAndDecode(frame)
                barcode_results = []

                if points is not None and len(decoded_info) > 0:
                    barcode_results.append({
                        "type": "QRCODE",
                        "data": decoded_info,
                        "points": points.astype(int).tolist()
                    })

                # 将图像转换为JPEG格式
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                return {
                    "frame": frame_base64,
                    "detected_objects": detected_objects,
                    "barcodes": barcode_results,
                    "fps": self.fps,
                    "status": "ok"
                }
        except Exception as e:
            print(f"Error in get_frame_data: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
        return {"status": "no_frame"}


# 在文件末尾添加API路由
scanner_app = None


# 添加登录路由
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if username in users and users[username]['password'] == password:
        return jsonify({
            "status": "success",
            "user": {
                "username": username,
                "role": users[username]['role']
            }
        })

    return jsonify({
        "status": "error",
        "message": "用户名或密码错误"
    }), 401


# 修改现有API，添加简单的会话验证
def check_auth():
    auth_header = request.headers.get('Authorization')
    return auth_header is not None


@app.route('/api/frame')
def get_frame():
    try:
        if not check_auth():
            return jsonify({"error": "Unauthorized"}), 401

        if scanner_app:
            frame_data = scanner_app.get_frame_data()
            if frame_data.get("status") == "ok":
                return jsonify(frame_data)
            elif frame_data.get("status") == "error":
                return jsonify({"error": frame_data["message"]}), 500
        return jsonify({"error": "Scanner not initialized"}), 503

    except Exception as e:
        print(f"Error in get_frame: {e}")
        return jsonify({"error": str(e)}), 500


# 添加错误处理装饰器
@app.errorhandler(Exception)
def handle_error(error):
    print(f"Unhandled error: {error}")
    return jsonify({"error": str(error)}), 500


@app.route('/api/history')
def get_history():
    if scanner_app:
        return jsonify(scanner_app.scan_history)
    return jsonify([])


@app.route('/api/heartbeat')
def heartbeat():
    return jsonify({"status": "alive"})


def run_flask():
    app.run(host='0.0.0.0', port=5000)


def main():
    global scanner_app
    # 创建主窗口
    root = tk.Tk()
    scanner_app = PackageScannerApp(root)

    # 启动Flask服务器
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    root.mainloop()


if __name__ == "__main__":
    main()