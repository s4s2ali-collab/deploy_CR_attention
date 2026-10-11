import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="AI Object Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Performance Enhancement: Cache the model load step
@st.cache_resource
def load_model():
    return YOLO("best.pt")


try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model 'best.pt'. Ensure the file is in the root directory. Details: {e}")
    st.stop()

# 3. Sidebar Configuration for Controls
st.sidebar.header("🔧 Model Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Technology Stack")
st.sidebar.info("YOLOv11 • Streamlit • OpenCV • PIL")

# 4. Main Page Header Layout
st.title("🔍 YOLO Object Detection Dashboard")
st.markdown(
    "Upload an image to detect objects using your custom-trained YOLO classifier model."
)
st.markdown("---")

# 5. File Uploader Layout
uploaded_file = st.file_uploader(
    "Choose an image file...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)

    # UI Feedback Spinner while processing
    with st.spinner("🧠 Analyzing image with YOLO model..."):
        # Corrected: Added [0] index to handle prediction array wrappers
        results = model.predict(source=image, conf=confidence_threshold)[0]
        
        result_image = results.plot()

        # Extract stats for metrics dashboard
        boxes = results.boxes
        total_detections = len(boxes)

        # Count classes found
        class_counts = {}
        names = model.names
        for box in boxes:
            cls_id = int(box.cls)
            class_name = names.get(cls_id, f"Class {cls_id}")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    # 6. High-Impact Metrics Section
    st.subheader("📊 Detection Analytics")
    col_metric1, col_metric2 = st.columns(2)
    with col_metric1:
        st.metric(label="Total Objects Detected", value=total_detections)
    with col_metric2:
        st.metric(label="Selected Confidence Limit", value=f"{confidence_threshold:.2f}")

    st.markdown("##")

    # 7. Side-by-Side Visual Comparison layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🎯 Detection Output")
        st.image(result_image, channels="BGR", use_container_width=True)

    # 8. Detailed breakdown list if objects exist
    if total_detections > 0:
        st.markdown("---")
        st.subheader("📝 Breakdown of Detected Items")
        breakdown_data = []
        for item, count in class_counts.items():
            breakdown_data.append(f"**{item}**: {count}")
        st.markdown("\n".join([f"- {i}" for i in breakdown_data]))
    else:
        st.info("ℹ️ No items crossed the current confidence threshold setting.")
else:
    # Beautiful empty state placeholder
    st.info("💡 Please upload an image file using the widget above to start.")
