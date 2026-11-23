import difflib
import easyocr
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st

st.set_page_config(page_title="Local OCR Screen Matcher", layout="wide")
st.title("Screen & Clipboard Data Matcher (No API Key Required)")
st.caption("Uses local EasyOCR + fuzzy text matching directly inside your Streamlit server.")

# Cache the OCR model so it only downloads/loads once per server boot
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)

with st.spinner("Loading OCR engine (takes ~10 seconds on first run)..."):
    reader = load_ocr_reader()

col1, col2 = st.columns(2)

with col1:
    copied_data = st.text_area(
        "1. Copied Clipboard Data (Paste with Ctrl+V / Cmd+V)",
        height=180,
        placeholder="Paste the text, invoice ID, or row you copied...",
    )

with col2:
    screenshot_file = st.file_uploader(
        "2. Upload or Paste Screenshot (Win+Shift+S / Cmd+Ctrl+Shift+4)",
        type=["png", "jpg", "jpeg"],
    )

match_threshold = st.slider(
    "Fuzzy Match Sensitivity (%)",
    min_value=50,
    max_value=100,
    value=80,
    help="100% requires an exact character-for-character match. Lower values catch typos or OCR quirks.",
)

if st.button("Scan Screen & Match Data", type="primary"):
    if not copied_data.strip() or not screenshot_file:
        st.warning("Please provide both your copied text and a screenshot.")
    else:
        with st.spinner("Reading text from screenshot..."):
            image = Image.open(screenshot_file).convert("RGB")
            img_array = np.array(image)

            # Extract text with bounding boxes: [([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], text, confidence), ...]
            ocr_results = reader.readtext(img_array)

            target = copied_data.strip().lower()
            full_screen_text = " ".join([item[1] for item in ocr_results])

            # Check 1: Direct substring match across combined screen text
            exact_found = target in full_screen_text.lower()

            # Check 2: Line-by-line / block-by-block fuzzy matching + bounding box highlight
            matches = []
            annotated_img = image.copy()
            draw = ImageDraw.Draw(annotated_img)

            for bbox, text, conf in ocr_results:
                clean_text = text.strip()
                # Compare similarity ratio
                ratio = difflib.SequenceMatcher(None, target, clean_text.lower()).ratio() * 100
                # Also boost if target is contained inside a longer detected block
                if target in clean_text.lower():
                    ratio = 100.0

                if ratio >= match_threshold:
                    matches.append((clean_text, round(ratio, 1), round(conf * 100, 1)))
                    # Draw green rectangle around matching text on screenshot
                    top_left = tuple(map(int, bbox[0]))
                    bottom_right = tuple(map(int, bbox[2]))
                    draw.rectangle([top_left, bottom_right], outline="#00FF00", width=4)

        st.divider()
        st.subheader("Match Results")

        if exact_found or matches:
            st.success(f"Match Found! ({len(matches)} matching text region(s) highlighted in green)")
        else:
            st.error("No Match Found above your sensitivity threshold.")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown("**Highlighted Screenshot**")
            st.image(annotated_img, use_container_width=True)

        with res_col2:
            st.markdown("**Matched Regions**")
            if matches:
                for m_text, sim, ocr_conf in sorted(matches, key=lambda x: x[1], reverse=True):
                    st.info(f"**Detected:** `{m_text}`  \nSimilarity: **{sim}%** | OCR Confidence: {ocr_conf}%")
            else:
                st.write("No individual text block met the threshold.")

            with st.expander("View All Raw Text Extracted From Screen"):
                st.text(full_screen_text if full_screen_text else "No text detected.")
