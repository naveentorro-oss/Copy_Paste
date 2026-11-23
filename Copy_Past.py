import difflib
import time

import easyocr
import mss
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="Live Screen Check", layout="wide")
st.title("Live Screen Check for Copied Data")
st.caption("Continuously checks a locally captured display for text matching your clipboard data using local OCR.")
st.info(
    "Screen capture runs on the machine hosting this Streamlit app. Run it locally to check your own desktop; "
    "a hosted deployment captures the server's display, not your browser's screen. Nothing is uploaded by this app."
)


@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)


try:
    with mss.mss() as capture:
        monitor_count = len(capture.monitors) - 1
except Exception as exc:
    st.error(f"Screen capture is unavailable: {exc}")
    st.stop()

monitor_options = {"All displays": 0}
monitor_options.update({f"Display {index}": index for index in range(1, monitor_count + 1)})

col1, col2 = st.columns(2)
with col1:
    copied_data = st.text_area(
        "Copied clipboard data",
        height=140,
        key="copied_data",
        placeholder="Paste the text, invoice ID, or row you want to find...",
    )
with col2:
    selected_monitor = st.selectbox(
        "Display to scan",
        options=list(monitor_options),
        help="Choose all displays or a single display on the machine running this app.",
    )

match_threshold = st.slider(
    "Fuzzy match sensitivity (%)",
    min_value=50,
    max_value=100,
    value=80,
    help="Lower values tolerate OCR errors and minor differences.",
)
scan_interval = st.slider("Scan interval (seconds)", min_value=1, max_value=10, value=2)
st.toggle("Start live screen checking", key="live_enabled")


@st.fragment(run_every=1)
def show_live_check():
    if not st.session_state.get("live_enabled", False):
        st.caption("Live checking is paused. Turn it on to start scanning the selected display.")
        return

    target = " ".join(st.session_state.get("copied_data", "").lower().split())
    if not target:
        st.warning("Paste the copied data above to start checking the screen.")
        return

    now = time.monotonic()
    last_scan = st.session_state.get("last_scan_time", 0.0)
    if now - last_scan < st.session_state.get("scan_interval", 2):
        result = st.session_state.get("last_scan_result")
        if result:
            render_result(result)
        else:
            st.info("Starting live screen check...")
        return

    st.session_state["last_scan_time"] = now
    monitor_index = monitor_options[st.session_state.get("selected_monitor", "All displays")]

    try:
        with mss.mss() as capture:
            screenshot = capture.grab(capture.monitors[monitor_index])
            image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

        reader = load_ocr_reader()
        ocr_results = reader.readtext(np.array(image))
        screen_text = " ".join(item[1] for item in ocr_results)
        normalized_screen_text = " ".join(screen_text.lower().split())
        exact_found = target in normalized_screen_text

        matches = []
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        for bbox, text, confidence in ocr_results:
            clean_text = text.strip()
            normalized_text = " ".join(clean_text.lower().split())
            if not normalized_text:
                continue
            similarity = difflib.SequenceMatcher(None, target, normalized_text).ratio() * 100
            if target in normalized_text:
                similarity = 100.0

            if similarity >= match_threshold:
                matches.append((clean_text, round(similarity, 1), round(confidence * 100, 1)))
                points = [tuple(map(int, point)) for point in bbox]
                draw.polygon(points, outline="#00FF00", width=4)

        result = {
            "image": annotated_image,
            "matches": matches,
            "exact_found": exact_found,
            "screen_text": screen_text,
            "scan_time": time.strftime("%H:%M:%S"),
        }
        st.session_state["last_scan_result"] = result
        render_result(result)
    except Exception as exc:
        st.error(f"Could not scan the display: {exc}")


def render_result(result):
    if result["exact_found"] or result["matches"]:
        st.success(f"Match found — last scanned at {result['scan_time']}.")
    else:
        st.error(f"No match found — last scanned at {result['scan_time']}.")

    image_col, text_col = st.columns(2)
    with image_col:
        st.markdown("**Live display (matching text outlined in green)**")
        st.image(result["image"], use_container_width=True)
    with text_col:
        st.markdown("**Matching text regions**")
        if result["matches"]:
            for detected, similarity, confidence in sorted(result["matches"], key=lambda item: item[1], reverse=True):
                st.info(f"**Detected:** `{detected}`  \nSimilarity: **{similarity}%** | OCR confidence: {confidence}%")
        else:
            st.write("No individual text region met the sensitivity threshold.")
        with st.expander("View all text extracted from the display"):
            st.text(result["screen_text"] or "No text detected.")


show_live_check()
