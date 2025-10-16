# 🖼️ Short Comic Panel Extractor (YOLOv5 + Streamlit)

An intelligent web application built with **Streamlit** and **YOLOv5** that automatically detects and extracts individual comic panels from a single-page comic image.

---

## 🌟 Features

- 🧠 **AI-powered detection** — uses a custom-trained YOLOv5 model to identify comic panel boundaries.  
- 📐 **Smart panel sorting** — orders panels automatically in correct reading order  
  (Left-to-right for Western comics or Right-to-left for Manga).  
- 📱 **Responsive design** — mobile-friendly layout using Streamlit’s flexible grid system.  
- 🖼️ **Instant preview and export** — visualize detected panels and download each as a separate image.  

---

## 🧩 Tech Stack

- [Streamlit](https://streamlit.io) — lightweight Python web app framework  
- [PyTorch](https://pytorch.org) — deep learning engine powering YOLOv5  
- [YOLOv5](https://github.com/ultralytics/yolov5) — object detection framework  
- [OpenCV](https://opencv.org) & [Pillow](https://python-pillow.org) — for image processing and manipulation  

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/short-comic-panel-extractor.git
cd short-comic-panel-extractor
