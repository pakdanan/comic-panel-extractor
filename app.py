import requests
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Comic Panel Splitter", layout="wide")

# =============================
# Utility: Sort panels function
# =============================
def sort_panels_by_column_then_row(items, rtl_order: bool) -> list:
    """
    Improved: Robust for multi-row layout (uneven panel counts per row)
    """
    if not items:
        return []

    data = []
    for item in items:
        if isinstance(item, np.ndarray):
            x, y, w, h = cv2.boundingRect(item)
        else:
            x, y, w, h = item
        data.append((item, x, y, w, h))

    data.sort(key=lambda d: d[2])  # sort by top Y

    rows = []
    row = [data[0]]
    y_threshold = data[0][4] * 0.3  # 30% height tolerance
    for i in range(1, len(data)):
        _, _, y, _, h = data[i]
        _, _, prev_y, _, prev_h = data[i-1]
        if abs(y - prev_y) < y_threshold:
            row.append(data[i])
        else:
            rows.append(row)
            row = [data[i]]
    rows.append(row)

    sorted_rows = []
    for row in rows:
        if rtl_order:
            sorted_rows.extend(sorted(row, key=lambda d: d[1], reverse=True))
        else:
            sorted_rows.extend(sorted(row, key=lambda d: d[1]))

    return [d[0] for d in sorted_rows]

# =============================
# Download model from Hugging Face (if needed) then load it
# =============================
MODEL_DIR = "ai-model"
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
HUGGINGFACE_MODEL_URL = "https://huggingface.co/mosesb/best-comic-panel-detection/resolve/main/best.pt"

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with st.spinner("📦 Downloading YOLO model from Hugging Face..."):
        response = requests.get(HUGGINGFACE_MODEL_URL, stream=True)
        total = int(response.headers.get("content-length", 0))
        with open(MODEL_PATH, "wb") as f:
            for data in response.iter_content(chunk_size=8192):
                f.write(data)

model = YOLO(MODEL_PATH)

# =============================
# Streamlit UI
# =============================
st.title("📚 Comic Panel Extractor")
st.markdown("Upload a comic page, and the app will detect and split each panel automatically.")

uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])
rtl_order = st.checkbox("Right-to-left reading order", value=False)

if uploaded_file:
    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Read and process
    image = cv2.imread(temp_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.subheader("🔍 Detection Results")

    # Run YOLOv8 inference
    results = model(image_rgb, conf=0.25)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    if len(boxes) == 0:
        st.warning("No panels detected. Try another image or adjust your model.")
    else:
        # Sort boxes
        panels = [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in boxes]
        sorted_panels = sort_panels_by_column_then_row(panels, rtl_order)

        # Draw boxes on image
        annotated = image_rgb.copy()
        for i, (x, y, w, h) in enumerate(sorted_panels):
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 3)
        st.image(annotated, caption="Detected Panels", use_column_width=True)

        st.subheader("🖼️ Extracted Panels")
        cols = st.columns(3)
        for i, (x, y, w, h) in enumerate(sorted_panels):
            panel_img = image_rgb[y:y+h, x:x+w]
            panel_pil = Image.fromarray(panel_img)
            with cols[i % 3]:
                st.image(panel_pil, caption=f"Panel {i+1}", use_column_width=True)
                st.download_button(
                    label=f"Download Panel {i+1}",
                    data=cv2.imencode('.png', cv2.cvtColor(np.array(panel_pil), cv2.COLOR_RGB2BGR))[1].tobytes(),
                    file_name=f"panel_{i+1}.png",
                    mime="image/png"
                )

    os.remove(temp_path)



