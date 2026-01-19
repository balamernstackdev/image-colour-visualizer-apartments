import streamlit as st
import numpy as np
import torch
import threading
import time
import pickle
from PIL import Image
import cv2
import os
import io
import logging
import requests
import gc
import warnings


# 🛡️ WARNING SHIELD: Silence technical chatter from AI libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# Also hide specific logs from noisy libraries
logging.getLogger("timm").setLevel(logging.ERROR)
logging.getLogger("mobile_sam").setLevel(logging.ERROR)

# 🚀 ULTIMATE 1GB RAM PROTECTION (Step Id 974+)
# Forcing single-threaded CPU mode prevents RAM spikes from massive stack allocations.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False # Save more RAM

# 🚀 DEPLOYMENT VERSION SYNC (Step Id 1714+)
# Increment this to force Streamlit Cloud to discard old AI logic caches.
CACHE_SALT = "V2.3.7-BALANCED-FIX"
APP_VERSION = "2.3.7"

from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_image_comparison import image_comparison
# 📦 DEPENDENCY GUARD
# We check these first to give the user a clean install instruction if they are missing locally.
try:
    import mobile_sam
    import timm
    from core.segmentation import SegmentationEngine, sam_model_registry
except Exception as e:
    import streamlit as st
    # TRANSIENT ERROR GUARD (Step Id 1597+)
    # Sometimes during deployment, Python's module system hits a "blip" (KeyError) 
    # while files are being moved. We catch this and show a friendly UI.
    st.error("### 🚀 Application Updating")
    st.info("The AI engine is syncing with the latest precision updates. This usually takes 2-3 seconds.")
    if "mobile_sam" in str(e) or "timm" in str(e):
        st.code("pip install timm git+https://github.com/ChaoningZhang/MobileSAM.git")
        st.write("If you are running locally, please run the command above.")
    else:
        st.write("Refreshing the environment. Please wait a moment...")
        st.button("Click to Refresh Now")
    st.stop()

# 🎯 GLOBAL AI CONFIGURATION
# 🎯 GLOBAL AI CONFIGURATION
MODEL_TYPE = "vit_t"
CHECKPOINT_PATH = "weights/mobile_sam.pt"

def ensure_weights():
    if not os.path.exists(CHECKPOINT_PATH):
        # --- INITIAL WEIGHTS DOWNLOADER (Dark Mode) ---
        download_placeholder = st.empty()
        download_placeholder.markdown("""
            <style>
            .download-overlay {
                position: fixed;
                top: 0; left: 0; width: 100vw; height: 100vh;
                background-color: #0E1117;
                display: flex; flex-direction: column; justify-content: center; align-items: center;
                z-index: 999999;
            }
            .download-spinner {
                border: 4px solid rgba(255, 255, 255, 0.1);
                border-radius: 50%;
                border-top: 4px solid #FF5A5F;
                width: 50px; height: 50px;
                animation: spin 1s linear infinite;
                margin-bottom: 20px;
            }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .download-text { color: white; font-family: sans-serif; font-size: 1.5rem; text-align: center;}
            </style>
            <div class="download-overlay">
                <div class="download-spinner"></div>
                <div class="download-text">Initializing AI Studio Engine...<br><small style='font-size: 0.9rem; color: #888;'>This one-time setup takes a few seconds.</small></div>
            </div>
        """, unsafe_allow_html=True)
        
        if not os.path.exists("weights"):
            os.makedirs("weights")
        url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(CHECKPOINT_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        download_placeholder.empty()
        st.success("AI Model Ready!")

ensure_weights()
from core.color_engine import ColorEngine
from core.quality import QualityEngine

# --- CONFIGURATION & STYLES ---
def setup_page():
    st.set_page_config(
        page_title="Color Visualizer Studio",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def setup_styles():
    st.markdown("""
        <style>
        /* IMPORT ROBOTO FONT (Professional Standard) */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

        /* RESET & BASE */
        :root {
            --primary: #FF5A5F;  /* Modern Coral Red */
            --bg-nav: #000000;
            --bg-main: #0E1117;
            --text-main: #FFFFFF;
            --card-shadow: 0 4px 12px rgba(0,0,0,0.5);
            --border-radius: 12px;
        }

        html, body, [data-testid="stAppViewContainer"], [data-testid="stMainComponent"], .main {
            background-color: var(--bg-main) !important;
            color: var(--text-main) !important;
            font-family: 'Roboto', sans-serif;
        }

        /* FORCE DARK STUDIO THEME */
        .stApp {
            background-color: var(--bg-main) !important;
        }

        /* HIDE DEFAULT STREAMLIT ELEMENTS BUT KEEP IN DOM */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stStatusWidget"] {display: none !important;} /* HIDE DEPLOY BUTTON SPECIFICALLY */
        
        header[data-testid="stHeader"] {
            z-index: 10 !important;
            background: transparent !important;
            pointer-events: none !important; /* Let clicks pass to my button */
        } 
        
        [data-testid="stToolbar"] {visibility: hidden !important; height: 0 !important;} 
        [data-testid="stDecoration"] {visibility: hidden !important; height: 0 !important;}

        /* FIXED SIDEBAR WIDTH & REMOVE COLLAPSE BUTTONS */
        [data-testid="stSidebarCollapseButton"] {
            display: none !important;
        }
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        section[data-testid="stSidebar"] {
            width: 400px !important;
            min-width: 400px !important;
            max-width: 400px !important;
        }
        
        /* BLACK THEME SIDEBAR */
        section[data-testid="stSidebar"] {
            background-color: #000000 !important;
            border-right: 1px solid #333333;
        }
        
        /* FIX WIDGET VISIBILITY & CONTRAST FOR DARK MODE */
        /* Force dark mode colors for all inputs/selects in sidebar */
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stTextInput > div > div, 
        [data-testid="stSidebar"] .stNumberInput > div > div {
            background-color: #1A1A1A !important;
            color: #FFFFFF !important;
            border: 1px solid #444444 !important;
        }

        /* FIX FILE UPLOADER BUTTON (Browse files) */
        [data-testid="stFileUploader"] button {
            background-color: #FF5A5F !important;
            color: white !important;
            border: none !important;
            border-radius: 20px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 500 !important;
        }
        /* Remove hover animation */
        [data-testid="stFileUploader"] button:hover {
            background-color: #FF5A5F !important;
        }

        /* FIX TOGGLE/CHECKBOX VISIBILITY */
        /* Ensure the toggle label is white and the track is visible */
        [data-testid="stCheckbox"] label, [data-testid="stToggle"] label {
            color: #FFFFFF !important; 
        }
        
        /* Style the Toggle Track (both ON and OFF states) */
        [data-testid="stToggle"] button {
            background-color: #444444 !important; /* Darker for dark mode */
            border: 2px solid #666666 !important;
            border-radius: 20px !important;
            min-width: 44px !important;
            height: 24px !important;
        }
        
        /* ON State - Bright coral/pink color */
        [data-testid="stToggle"] button[aria-checked="true"] {
            background-color: #FF5A5F !important;
            border-color: #FF5A5F !important;
        }
        
        /* Ensure the toggle handle/thumb is visible */
        [data-testid="stToggle"] button > div {
            background-color: #FFFFFF !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.4) !important;
            width: 18px !important;
            height: 18px !important;
            border-radius: 50% !important;
            border: 1px solid #ccc !important;
        }
        
        /* Make sure the clickable area is actually clickable */
        [data-testid="stCheckbox"], [data-testid="stToggle"] {
            z-index: 100 !important;
        }
        
        /* FIX COLOR PICKER VISIBILITY */
        [data-testid="stColorPicker"] > div > div {
             border: 2px solid #444444 !important;
             border-radius: 50% !important;
             box-shadow: 0 2px 5px rgba(0,0,0,0.4) !important;
             transform: scale(1.1) !important;
        }
        /* Ensure the color value is visible */
        [data-testid="stColorPicker"] input {
            font-weight: bold !important;
            color: #FFFFFF !important;
            background-color: #1A1A1A !important;
        }

        /* FIX SYSTEM SYNC EXPANDER & BUTTONS */
        .stExpander {
            border: 1px solid #333333 !important;
            background-color: #000000 !important;
            border-radius: 8px !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Sidebar Buttons (Secondary style) */
        [data-testid="stSidebar"] button[kind="secondary"] {
            background-color: #222222 !important;
            color: #FFFFFF !important;
            border: 1px solid #444444 !important;
            border-radius: 8px !important;
        }

        /* PRIMARY BUTTONS (Pink Gradient) */
        button[kind="primary"] {
            background: linear-gradient(135deg, #FF5A5F 0%, #FF8084 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 30px !important;
            box-shadow: 0 4px 10px rgba(255, 90, 95, 0.2) !important;
            font-weight: 600 !important;
        }
        
        /* Text Color inside Inputs */
        [data-testid="stSidebar"] [data-baseweb="select"] span {
            color: #FFFFFF !important;
        }

        /* FORCE SIDEBAR TEXT WHITE */
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3, 
        section[data-testid="stSidebar"] p, 
        section[data-testid="stSidebar"] span, 
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] li,
        section[data-testid="stSidebar"] small {
            color: #FFFFFF !important;
        }
        
        /* GLOBAL DARK MODE OVERRIDES */
        .main, [data-testid="stAppViewContainer"], [data-testid="stMainComponent"] {
            background-color: #0E1117 !important;
        }
        
        /* FORCE ALL TEXT IN MAIN AREA TO LIGHT */
        .main p, .main label, .main span, .main div, .main small {
            color: #FFFFFF !important;
        }

        /* HEADERS */
        h1, h2, h3 {
            font-weight: 700;
            color: #FFFFFF !important;
            letter-spacing: -0.5px;
        }
        
        /* SPINNER/LOADER */
        .stSpinner > div {
            border-top-color: #FF5A5F !important;
        }

        /* TOAST NOTIF */
        .stToast {
            background-color: #333 !important;
            color: white !important;
            border-radius: 8px;
        }
        
        /* COLOR PICKER CIRCLE */
        input[type="color"] {
            border-radius: 50%;
            width: 40px;
            height: 40px;
            border: 2px solid #ddd;
            padding: 0;
            overflow: hidden;
            cursor: pointer;
        }
        
        /* FILE UPLOADER TEXT IN SIDEBAR */
        [data-testid="stSidebar"] [data-testid="stFileUploader"] p,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] small,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] span {
            color: #BBBBBB !important;
        }

        /* RADIO BUTTONS IN SIDEBAR */
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            color: #FFFFFF !important;
        }
        
        /* Ensure object name text is white and readable */
        [data-testid="stSidebar"] button {
            color: #FFFFFF !important;
        }
        
        /* Selectbox item text on dark background */
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #1A1A1A !important;
            color: white !important;
        }
        
        /* SIDEBAR ALERTS (INFO/ERROR) */
        [data-testid="stSidebar"] [data-testid="stNotification"] {
            background-color: #111111 !important;
            color: white !important;
            border: 1px solid #333333 !important;
        }
        [data-testid="stSidebar"] [data-testid="stNotificationContent"] p {
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

# --- MODEL MANAGEMENT ---
@st.cache_resource
def get_sam_model(path, type_name, salt=""):
    """Load and cache the heavy model weights globally."""
    if not os.path.exists(path):
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create the model instance using the registry directly
    model = sam_model_registry[type_name](checkpoint=path)
    model.to(device=device)
    # We don't set model.device here to avoid AttributeError on cloud environments
    return model

@st.cache_resource
def get_global_lock():
    """Lock to prevent multiple AI sessions from hitting the CPU at once."""
    return threading.Lock()

@st.cache_resource
def get_sam_engine_singleton(checkpoint_path, model_type, salt=""):
    """Global engine singleton to avoid session state duplication."""
    model = get_sam_model(checkpoint_path, model_type, salt=salt)
    if model is None:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SegmentationEngine(model_instance=model, device=device)

def get_sam_engine(checkpoint_path, model_type):
    """Wrapped getter."""
    return get_sam_engine_singleton(checkpoint_path, model_type, salt=CACHE_SALT)

# --- HELPER LOGIC ---
def get_crop_params(image_width, image_height, zoom_level, pan_x, pan_y):
    """
    Calculate crop coordinates based on zoom and normalized pan (0.0 to 1.0).
    """
    if zoom_level <= 1.0:
        return 0, 0, image_width, image_height

    # visible width/height
    view_w = int(image_width / zoom_level)
    view_h = int(image_height / zoom_level)

    # ensure pan stays within bounds
    # pan_x=0 -> left edge, pan_x=1 -> right edge
    max_x = image_width - view_w
    max_y = image_height - view_h
    
    start_x = int(max_x * pan_x)
    start_y = int(max_y * pan_y)
    
    return start_x, start_y, view_w, view_h

def composite_image(original_rgb, masks_data):
    """
    Apply all color layers to user image using optimized multi-layer blending.
    PERFORMANCE: Uses incremental background caching to speed up tuning.
    """
    # Filter only visible layers
    visible_masks = [m for m in masks_data if m.get('visible', True)]
    
    # Base Image
    base = original_rgb.copy()
    
    # Process each layer
    # Note: We simplified caching for now to ensure correctness with the new engine.
    # Refined caching can be re-added if performance drops.
    
    # Process each layer
    # OPTIMIZATION: Use incremental background caching
    # If we have N layers, and we simply changed properties of layer N, 
    # we can reuse the cached composite of layers 0..N-1.
    
    # Check cache validity
    cache = st.session_state.get("bg_cache")
    start_index = 0
    
    # Validation: 
    # 1. We have a cache
    # 2. We have at least 1 mask
    # 3. Cache corresponds to the first N-1 masks in the current list (deep check)
    if (len(visible_masks) > 1 and cache is not None and 
        cache.get("mask_count") == len(visible_masks) - 1):
        
        # Verify cached masks match current start of list
        # We check IDs or content signatures? 
        # Checking IDs of mask arrays is fast.
        valid_cache = True
        
        if cache['image'].shape[:2] != original_rgb.shape[:2]:
            valid_cache = False
        else:
            for i in range(len(visible_masks) - 1):
                # If the 'mask' object is the same instance, it's likely unchanged?
                # But 'opacity' etc are dict keys.
                # We need to check if the dict content changed.
                curr = visible_masks[i]
                cached_meta = cache['meta'][i] # We need to store meta in cache
                
                # Compare critical keys
                if (curr.get('color') != cached_meta.get('color') or 
                    curr.get('opacity') != cached_meta.get('opacity') or
                    curr.get('finish') != cached_meta.get('finish')):
                    valid_cache = False
                    break
                    
        if valid_cache:
            base = cache['image'].copy()
            start_index = len(visible_masks) - 1 # Only process the last one
    
    # Render loop
    for i in range(start_index, len(visible_masks)):
        layer = visible_masks[i]
        
        # 1. Refine Edge (Quality Engine)
        if 'mask_refined' not in layer:
            layer['mask_refined'] = QualityEngine.refine_edge(original_rgb, layer['mask'])
            
        # 2. Apply Material
        paint_layer = ColorEngine.apply_material(
            original_rgb,
            layer['mask_refined'],
            layer.get('color', '#FF0000'), 
            finish=layer.get('finish', 'Matte'),
            intensity=layer.get('opacity', 1.0)
        )
        
        # 3. Composite
        mask_alpha = layer['mask_refined']
        if len(mask_alpha.shape) == 2:
            mask_alpha = mask_alpha[:,:,None]
            
        base = base * (1.0 - mask_alpha) + paint_layer * mask_alpha
        
        # Update Cache if this is the N-1 layer
        if len(visible_masks) > 1 and i == len(visible_masks) - 2:
             st.session_state["bg_cache"] = {
                "image": base.copy(), # Snapshot of layers 0..N-1
                "mask_count": len(visible_masks) - 1,
                "meta": [m.copy() for m in visible_masks[:-1]] # Store metadata for validation
             }
             
    return base.astype(np.uint8)

# --- IMAGE PROCESSING HELPER ---
def process_uploaded_file(uploaded_file, sam):
    """Centralized logic to load and initialize image state."""
    if uploaded_file is None:
        return
        
    try:
        # Avoid redundant reloads
        if st.session_state.get("image_path") == uploaded_file.name and st.session_state.get("image") is not None:
            return

        # 1. Decode Image
        file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        if image is None:
            st.error("Failed to decode image. Please ensure it is a valid JPG or PNG.")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Store Original
        st.session_state["image_original"] = image.copy()
        
        # 3. Create Work Image (Preview)
        # PERFORMANCE: 720px is sweet spot for CPU speed vs AI precision
        max_dim = 720 
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        st.session_state["image"] = image
        st.session_state["image_path"] = uploaded_file.name
        st.session_state["masks"] = []
        st.session_state["composited_cache"] = image.copy() 
        
        # 4. Clear Cache
        for k in list(st.session_state.keys()):
            if k.startswith("base_l_"):
                del st.session_state[k]
        
        # 5. Reset SAM
        if sam:
            sam.is_image_set = False
        st.session_state["ai_ready"] = False
        
        st.toast("✅ Image Loaded Successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return

def initialize_session_state():
    """Initialize all session state variables with multi-layer safety."""
    defaults = {
        "image": None,          # 640px preview image
        "image_original": None, # Full resolution original
        "file_name": None,
        "masks": [],
        "zoom_level": 1.0,
        "pan_x": 0.5,
        "pan_y": 0.5,
        "last_click_global": None,
        "mask_level": None, # 0, 1, or 2 for granularity
        "bg_cache": None,   # For selective compositing performance
        "sampling_mode": False,
        "composited_cache": None,
        "render_id": 0,         # Versioning for instant UI refresh
        "picked_color": "#ff4b4b", # Default primary color
        "engine_ready": False,  # Prevent redundant loading spinners
        "selected_layer_idx": None, # Track which layer is being edited
        "last_export": None,     # High-res download buffer
        "active_selection": None # NEW: Temporary holding area for current segmentation
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Extra Safety: Ensure all current masks have the latest properties 
    if st.session_state.get("masks"):
        for m in st.session_state["masks"]:
            if 'visible' not in m: m['visible'] = True
            if 'brightness' not in m: m['brightness'] = 0.0
            if 'contrast' not in m: m['contrast'] = 1.0
            if 'saturation' not in m: m['saturation'] = 1.0
            if 'hue' not in m: m['hue'] = 0.0
            if 'opacity' not in m: m['opacity'] = 1.0
            if 'finish' not in m: m['finish'] = 'Standard'
            if 'tex_rot' not in m: m['tex_rot'] = 0
            if 'tex_scale' not in m: m['tex_scale'] = 1.0
            # CRITICAL: Clear heavy cached blur arrays to save RAM
            if 'mask_soft' in m: del m['mask_soft']

# --- UI COMPONENTS ---
def render_sidebar(sam, device_str, composite_full=None):
    with st.sidebar:
        st.title("🎨 Visualizer Studio")
        st.caption(f"App Version: {APP_VERSION}")
        st.caption(f"AI Engine: {device_str}")
        
        # Upload Section
        # CRITICAL: Must use a FIXED KEY to prevent widget reset when the spinner above disappears
        uploaded_file = st.file_uploader("Start Project", type=["jpg", "png", "jpeg"], label_visibility="collapsed", key="project_uploader")
        
        if uploaded_file is not None:
            process_uploaded_file(uploaded_file, sam)
        else:
            # Reset state if file cleared
            if st.session_state.get("image") is not None:
                st.session_state["image"] = None
                st.session_state["image_original"] = None
                st.session_state["image_path"] = None
                st.session_state["masks"] = []
                st.session_state["composited_cache"] = None
                if sam: sam.is_image_set = False
                st.rerun()
        
        # Project Persistence
        if st.session_state.get("image") is not None:
            st.divider()
            st.subheader("💾 Save & Load")
            
            # 1. Save Project
            project_data = {
                "masks": st.session_state["masks"],
                "image_path": st.session_state.get("image_path")
            }
            project_bytes = pickle.dumps(project_data)
            
            st.download_button(
                label="� Download Project",
                data=project_bytes,
                file_name=f"{st.session_state.get('image_path', 'project').split('.')[0]}.studio",
                mime="application/octet-stream",
                use_container_width=True,
                help="Save your painted layers to resume later"
            )
            
            # 2. Load Project
            st.caption("Load Saved Project")
            loaded_proj = st.file_uploader(
                "Upload Project File", 
                type=["studio"], 
                label_visibility="collapsed",
                key="project_loader"
            )
            if loaded_proj is not None:
                if st.button("� Load Project", use_container_width=True):
                    try:
                        data = pickle.loads(loaded_proj.read())
                        if data.get("image_path") != st.session_state.get("image_path"):
                            st.warning("⚠️ Project may not match this image!")
                        
                        # Ensure all loaded masks have modern keys (back-compat)
                        loaded_masks = data["masks"]
                        for m in loaded_masks:
                            if 'visible' not in m: m['visible'] = True
                            if 'brightness' not in m: m['brightness'] = 0.0
                            if 'contrast' not in m: m['contrast'] = 1.0
                            if 'saturation' not in m: m['saturation'] = 1.0
                            if 'hue' not in m: m['hue'] = 0.0
                        
                        st.session_state["masks"] = loaded_masks
                        st.session_state["composited_cache"] = None
                        st.success("✅ Project Loaded!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to load: {e}")
            
            # --- COMPARISON MODE TOGGLE ---
            st.divider()
            show_comparison = st.checkbox(
                "📊 Compare Original",
                value=st.session_state.get("show_comparison", False),
                help="View before/after comparison. Editing is disabled in comparison mode."
            )
            
            if show_comparison != st.session_state.get("show_comparison", False):
                st.session_state["show_comparison"] = show_comparison
                st.rerun()

    
        # --- ACTIVE SELECTION HUB (New Workflow) ---
        st.divider()
        
        # --- GLOBAL COLOR PICKER (Moved Outside v1.6.1) ---
        st.subheader("🎨 Color Palette")
        
        preset_colors = {
            "White": "#FFFFFF", "Off White": "#FAF9F6", "Cream": "#FFFDD0", 
            "Sage Green": "#8FBC8F", "Sky Blue": "#87CEEB", "Lavender": "#E6E6FA", 
            "Peach": "#FFDAB9", "Terracotta": "#E2725B", "Mustard": "#FFDB58",
            "Forest Green": "#228B22", "Royal Blue": "#4169E1", "Crimson": "#DC143C",
            "Midnight Navy": "#000080", "Charcoal": "#36454F", "Teal": "#008080", "Gold": "#FFD700"
        }
        
        
        # 1. Dropdown Color Selection (Replaced visual swatches per user request)
        color_names = list(preset_colors.keys())
        
        # Find current color name if it matches a preset
        current_color = st.session_state.get("picked_color", "#FFFFFF")
        try:
            current_index = list(preset_colors.values()).index(current_color)
        except ValueError:
            current_index = 0  # Default to White if custom color
        
        selected_color_name = st.selectbox(
            "Choose Color",
            color_names,
            index=current_index,
            key="color_dropdown"
        )
        
        # Automatically update picked color when dropdown changes
        selected_hex = preset_colors[selected_color_name]
        
        # Check if dropdown changed the color
        if selected_hex != st.session_state.get("picked_color"):
            st.session_state["picked_color"] = selected_hex
            # FORCE UPDATE the color picker widget's key
            st.session_state["global_picker"] = selected_hex
            st.rerun()

        # 2. Custom Color Picker - now synced with dropdown
        picked_color = st.color_picker(
            "Custom Palette", 
            st.session_state.get("picked_color", "#FFFFFF"), 
            key="global_picker"
        )
        
        # Update picked_color when user manually changes the picker
        if picked_color != st.session_state.get("picked_color"):
             # If user manually picked a color that doesn't match current state
             st.session_state["picked_color"] = picked_color
             # We don't need to rerun here, the widget update handles it naturally
        
        # GLOBAL FINISH (For Instant Mode) - Moved below Color for logical flow
        # We need a persistent finish choice
        if st.session_state.get("instant_mode"):
             st.caption("Material")
             global_finish = st.selectbox(
                 "Finish (Shine)", 
                 ["Matte", "Satin", "Glossy"], 
                 index=0, 
                 key="global_finish_pref", 
                 help="Determine how shiny the paint looks.\nMatte = Flat\nGlossy = Shiny"
             )
        
        st.divider()
        
        if st.session_state.get("active_selection"):
            st.subheader("✨ Active Object")
            st.caption("Step 2: Apply Color")

            # Finish Selector (Immediate Preview)
            st.selectbox(
                "Finish", 
                ["Matte", "Satin", "Glossy"], 
                index=0, 
                key="active_finish",
                help="Matte: Flat, dry look.\nSatin: Soft sheen.\nGlossy: Shiny, reflective."
            )
            
            # --- SELECTION REFINEMENT MODE (Restored v1.4.3) ---
            with st.container():
                st.caption("🛠️ Actions")
                # Edit Mode: New vs Refine
                # Renamed 'Refine Last Layer' to 'Multi-Select / Refine' to clarify creating complex selections.
                edit_mode = st.radio(
                    "Mode", 
                    ["New Object", "Multi-Select / Refine"], 
                    horizontal=True,
                    help="**New Object:** Starts a fresh selection.\n\n**Multi-Select / Refine:** Add more parts to the current selection (e.g. select sofa + chair) or fix mistakes."
                )
                st.session_state["paint_mode"] = edit_mode # Sync with paint_mode for consistency
                
                if edit_mode == "Multi-Select / Refine":
                    refine_type = st.radio(
                        "Action", 
                        ["➕ Add Area", "➖ Remove Area"], 
                        index=0, 
                        horizontal=True,
                        key="active_refine_mode"
                    )
                    st.session_state["click_label"] = 1 if "Add" in refine_type else 0
                else: # New Object mode
                    st.session_state["click_label"] = 1 # Always add for new object
                    
                    # CRITICAL FIX: If user switches to "New Object", DESELECT any existing layer
                    # so that changing the color doesn't edit the old object.
                    if st.session_state.get("selected_layer_idx") is not None:
                        st.session_state["selected_layer_idx"] = None
                        st.rerun()
            
            # CRITICAL: Instant Preview Update
            # Sync the color to the Active Selection
            if st.session_state.get("active_selection"):
                 st.session_state["active_selection"]["color"] = picked_color

            # Preview is handled in the Main Loop for performance and correct z-ordering.
            
            # ACTIONS (MOVED TO MAIN AREA v2.4)
            # st.caption("🛠️ Actions")
            # col_apply, col_discard = st.columns(2)
            # ... (moved to render_action_bar)
            pass
        
            st.divider()
            st.info("💡 Tip: Click more points on the image to refine this object before applying.")

        else:
            # No Selection - Standard Mode
            st.info("Click on a wall, floor, or object in the image to start.")
            
            # Painting Mode Control (Only if layers exist)
            if st.session_state["masks"]:
                st.caption("Mode")
                options = ["New Object", "Refine Last Layer"]
                mode = st.radio("Mode", options, index=0, label_visibility="collapsed", key="paint_mode_selector")
                # Old Refine Block Removed.

            else:
                 st.session_state["paint_mode"] = "New Object"

        st.divider()
        st.subheader("⚡ Workflow")
        st.toggle("Instant Paint Mode (One-Click)", key="instant_mode", help="Skip confirmation. Clicks immediately paint the wall.")
        
        # Segmentation Control (Exposed)
        st.divider()
        # Segmentation Control (Exposed)
        st.divider()
        st.subheader("🧠 Intelligence")
        
        # 1. Mode Selection
        sens_mode = st.radio(
            "Target Mode", 
            ["🎯 Small Details", "🧱 Walls & Floors", "🛋️ Furniture"], 
            index=0,
            horizontal=False,
            help="**Small Details:** Trim, handles, switches. Best start.\n\n**Walls & Floors:** Smart separation. Fills walls but respects corners & holes.\n\n**Furniture:** Best for sofas, tables, cabinets."
        )
        
        # Map Selection to AI Level
        if "Walls" in sens_mode:
            st.session_state["mask_level"] = 2
        elif "Furniture" in sens_mode:
            st.session_state["mask_level"] = 1 # Intermediate level
        else: # Small Details
            st.session_state["mask_level"] = 0
            
        # 2. Texture Guard Toggle (Manual Override)
        use_texture_guard = st.toggle("🛡️ Protect Details (TVs/Art)", value=True, help="Prevents painting over detailed objects like TV screens or paintings.")
        st.session_state["use_texture_guard"] = use_texture_guard 



        # --- SYSTEM RECOVERY ---
        with st.sidebar.expander("🛠️ System Sync"):
            st.caption(f"Backend Version: {CACHE_SALT}")
            if st.button("🔄 Hard Reset AI Engine", use_container_width=True, help="Forcefully clears the AI cache and reloads the latest precision logic."):
                st.cache_resource.clear()
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        
        # Lighting & Adjustments (Hidden to avoid interruption)
        # if st.session_state["masks"]:
        #     st.divider()
        #     st.subheader("💡 Layer Tuning")
        #     
        #     # Sync with selection hub
        #     idx = st.session_state.get("selected_layer_idx")
        #     if idx is not None and 0 <= idx < len(st.session_state["masks"]):
        #         target_mask = st.session_state["masks"][idx]
        #     else:
        #         target_mask = st.session_state["masks"][-1]
        #     
        #     # Opacity
        #     op = st.slider("Opacity", 0.0, 1.0, target_mask.get("opacity", 1.0), key="tune_opacity")
        #     if op != target_mask.get("opacity", 1.0):
        #         target_mask["opacity"] = op
        #         st.session_state["bg_cache"] = None # Invalidate cache
        #         st.session_state["composited_cache"] = None
        #         st.rerun()
        # 
        #     # Finish
        #     finish = st.selectbox("Finish", ["Standard", "Matte", "Glossy"], index=["Standard", "Matte", "Glossy"].index(target_mask.get("finish", "Standard")), key="tune_finish")
        #     if finish != target_mask.get("finish", "Standard"):
        #         target_mask["finish"] = finish
        #         st.session_state["bg_cache"] = None
        #         st.session_state["composited_cache"] = None
        #         st.rerun()
        #         
        #     with st.expander("Advanced color"):
        #         # Brightness
        #         br = st.slider("Brightness", -0.5, 0.5, target_mask.get("brightness", 0.0), key="tune_bright")
        #         if br != target_mask.get("brightness", 0.0):
        #             target_mask["brightness"] = br
        #             st.session_state["bg_cache"] = None
        #             st.session_state["composited_cache"] = None
        #             st.rerun()
        #             
        #         # Contrast
        #         ct = st.slider("Contrast", 0.5, 1.5, target_mask.get("contrast", 1.0), key="tune_contrast")
        #         if ct != target_mask.get("contrast", 1.0):
        #             target_mask["contrast"] = ct
        #             st.session_state["bg_cache"] = None
        #             st.session_state["composited_cache"] = None
        #             st.rerun()
        #             
        #         # Saturation
        #         sat = st.slider("Saturation", 0.0, 3.0, target_mask.get("saturation", 1.0), key="tune_sat")
        #         if sat != target_mask.get("saturation", 1.0):
        #             target_mask["saturation"] = sat
        #             st.session_state["bg_cache"] = None
        #             st.session_state["composited_cache"] = None
        #             st.rerun()
        # 
        #         # Hue
        #         hue = st.slider("Hue Shift", -180, 180, int(target_mask.get("hue", 0)), key="tune_hue")
        #         if hue != target_mask.get("hue", 0):
        #             target_mask["hue"] = float(hue)
        #             st.session_state["bg_cache"] = None
        #             st.session_state["composited_cache"] = None
        #             st.rerun()
        
        # --- OBJECT MANAGER (v2.0 Sidebar) ---
        st.divider()
        st.subheader("🏗️ Room Objects")
        
        has_layers = len(st.session_state["masks"]) > 0
        
        if has_layers:
            # Render Object List using simplified Cards
            for i, mask_data in enumerate(st.session_state["masks"]):
                
                # Active Indicator
                is_selected = st.session_state.get("selected_layer_idx") == i
                
                # Create a card for each object
                # Use border=True for distinct separation (Mocking a list item)
                with st.container(border=True):
                     # Optimized Columns: Vis(1) | Name(3) | Finish(2) | Delete(1)
                     c1, c2, c3, c4 = st.columns([0.6, 3.0, 2.0, 0.8], vertical_alignment="center")
                     
                     # Visibility Toggle
                     with c1:
                         vis_icon = "👁️" if mask_data.get('visible', True) else "🚫"
                         # Minimal button to save space
                         if st.button(vis_icon, key=f"vis_btn_{i}", help="Toggle Visibility"):
                             mask_data['visible'] = not mask_data.get('visible', True)
                             st.session_state["composited_cache"] = None
                             st.rerun()

                     # Object Name (Clickable)
                     with c2:
                         display_name = f"{i+1}. {mask_data.get('name', 'Object')}"
                         # Primary button for selected, Secondary for others
                         btn_type = "primary" if is_selected else "secondary" 
                         if st.button(display_name, key=f"sel_btn_{i}", use_container_width=True, type=btn_type):
                             st.session_state["selected_layer_idx"] = i
                             st.session_state["active_selection"] = None
                             if mask_data.get("color"):
                                 st.session_state["picked_color"] = mask_data["color"]
                             st.rerun()
                     
                     # Material Selector (Finish)
                     with c3:
                         current_finish = mask_data.get('finish', 'Matte')
                         new_finish = st.selectbox(
                             "Finish", 
                             ["Matte", "Satin", "Glossy"], 
                             index=["Matte", "Satin", "Glossy"].index(current_finish),
                             key=f"finish_sel_{i}",
                             label_visibility="collapsed"
                         )
                         if new_finish != current_finish:
                             mask_data['finish'] = new_finish
                             st.session_state["composited_cache"] = None
                             st.rerun()

                     # Delete Action
                     with c4:
                         if st.button("🗑️", key=f"del_btn_{i}", help="Remove Object"):
                             with st.spinner("Removing..."):
                                 st.session_state["masks"].pop(i)
                                 if st.session_state.get("selected_layer_idx") == i:
                                     st.session_state["selected_layer_idx"] = None
                                 
                                 st.session_state["composited_cache"] = None
                                 st.session_state["bg_cache"] = None
                                 st.session_state["render_id"] += 1
                                 st.rerun()
            
            # Global Actions
            if st.button("🗑️ Clear All Objects", use_container_width=True):
                 st.session_state["masks"] = []
                 st.session_state["selected_layer_idx"] = None
                 st.session_state["composited_cache"] = None
                 st.session_state["bg_cache"] = None
                 st.session_state["render_id"] += 1
                 st.rerun()

        else:
            st.info("👆 Click on the image to detect your first object!")

        # Download Section
        st.divider()
        if st.session_state["image"] is not None and st.session_state["masks"]:
            if st.button("💎 Prepare High-Res Download", use_container_width=True, help="Processes your design at original resolution for maximum quality."):
                with st.spinner("Processing 4K Export..."):
                    try:
                        # Scale Masks to Original Resolution
                        original_img = st.session_state["image_original"]
                        oh, ow = original_img.shape[:2]
                        
                        # Create High-Res Mask list
                        high_res_masks = []
                        for m_data in st.session_state["masks"]:
                            hr_m = m_data.copy()
                            # Scale the actual boolean mask
                            mask_uint8 = (m_data['mask'] * 255).astype(np.uint8)
                            hr_mask_uint8 = cv2.resize(mask_uint8, (ow, oh), interpolation=cv2.INTER_LINEAR)
                            hr_m['mask'] = hr_mask_uint8 > 127
                            hr_m['mask_soft'] = None 
                            high_res_masks.append(hr_m)
                        
                        # Render High-Res Composite
                        # We must manually iterate and blend since ColorEngine works per-layer
                        
                        # 1. Start with Original Image
                        final_composite = original_img.copy().astype(np.float32) / 255.0
                        
                        for m_data in high_res_masks:
                            # 2. Apply Material to Original (for correct texture)
                            # We use m_data['mask'] which is the high-res boolean mask
                            
                            layer_img = ColorEngine.apply_material(
                                original_img,
                                m_data['mask'], 
                                m_data.get('color', '#FFFFFF'),
                                finish=m_data.get('finish', 'Matte'),
                                intensity=m_data.get('opacity', 1.0)
                            )
                            layer_float = layer_img.astype(np.float32) / 255.0
                            
                            # 3. Alpha Blend
                            mask_f = m_data['mask'].astype(np.float32)
                            if len(mask_f.shape) == 2: mask_f = mask_f[:,:,None]
                            
                            # Standard Alpha Blending: Result = Base * (1-alpha) + Layer * alpha
                            final_composite = final_composite * (1.0 - mask_f) + layer_float * mask_f
                            
                        # Convert back to uint8
                        dl_comp = (np.clip(final_composite, 0, 1) * 255).astype(np.uint8)
                        dl_pil = Image.fromarray(dl_comp)
                        dl_buf = io.BytesIO()
                        dl_pil.save(dl_buf, format="PNG")
                        st.session_state["last_export"] = dl_buf.getvalue()
                        st.success("✅ Download Ready!")
                    except Exception as e:
                        st.error(f"Export failed: {e}")

            if st.session_state.get("last_export"):
                st.download_button(
                    label="📥 Save Final Image",
                    data=st.session_state["last_export"],
                    file_name="pro_visualizer_design.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            
            # --- SYNC SELECTED LAYER TO PICKER ---
            # If a layer is selected, and the picker changed, update the layer.
            if st.session_state.get("selected_layer_idx") is not None:
                idx = st.session_state["selected_layer_idx"]
                if 0 <= idx < len(st.session_state["masks"]):
                    target_mask = st.session_state["masks"][idx]
                    # Only update if different (avoid redundant cache clears)
                    if target_mask.get("color") != st.session_state["picked_color"]:
                        target_mask["color"] = st.session_state["picked_color"]
                        # Auto-Update Name if it's generic? No, keep name.
                        
                        st.session_state["composited_cache"] = None
                        st.session_state["bg_cache"] = None # Color changed -> background cache invalid
                        st.session_state["render_id"] += 1
                        st.rerun()

        return # No longer need to return picked_color

def overlay_pan_controls(image):
    """Draws semi-transparent pan arrows on the image edges."""
    h, w, c = image.shape
    overlay = image.copy()
    
    # Define color (white with transparency) and thickness
    color = (255, 255, 255)
    thickness = 2
    
    # Margin specific to display size
    margin = 40
    center_x, center_y = w // 2, h // 2
    
    # Draw Arrows
    # Top Arrow
    cv2.arrowedLine(overlay, (center_x, margin), (center_x, 10), color, thickness, tipLength=0.5)
    
    # Bottom Arrow
    cv2.arrowedLine(overlay, (center_x, h - margin), (center_x, h - 10), color, thickness, tipLength=0.5)
    
    # Left Arrow
    cv2.arrowedLine(overlay, (margin, center_y), (10, center_y), color, thickness, tipLength=0.5)
    
    # Right Arrow
    cv2.arrowedLine(overlay, (w - margin, center_y), (w - 10, center_y), color, thickness, tipLength=0.5)
    
    # Blend overlay
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image

def render_zoom_controls():
    """Render zoom and pan controls below the image."""
    
    # Shared Column Ratio for Alignment
    # Spacer(5) | LeftBtn(1) | Center(1.5) | RightBtn(1) | Spacer(5)
    ratio = [5, 0.8, 1.2, 0.8, 5]

    # --- Zoom Row ---
    z_col1, z_col2, z_col3, z_col4, z_col5 = st.columns(ratio, vertical_alignment="center")
    
    def update_zoom(delta):
        st.session_state["zoom_level"] = max(1.0, min(4.0, st.session_state["zoom_level"] + delta))
    
    with z_col2:
        if st.button("➖", help="Zoom Out", use_container_width=True):
            update_zoom(-0.2)
            st.rerun()
            
    with z_col3:
        st.markdown(
            f"""
            <div style='
                text-align: center; 
                font-weight: bold; 
                background-color: #f0f2f6; 
                color: #31333F;
                padding: 6px 10px; 
                border-radius: 4px;
                border: 1px solid #dcdcdc;
            '>
                {int(st.session_state['zoom_level'] * 100)}%
            </div>
            """, 
            unsafe_allow_html=True
        )
            
    with z_col4:
        if st.button("➕", help="Zoom In", use_container_width=True):
            update_zoom(0.2)
            st.rerun()

    # --- Reset Button ---
    if st.session_state["zoom_level"] > 1.0 or st.session_state["pan_x"] != 0.5 or st.session_state["pan_y"] != 0.5:
        r_col1, r_col2, r_col3 = st.columns([4, 2, 4])
        with r_col2:
            if st.button("🎯 Reset View", use_container_width=True):
                st.session_state["zoom_level"] = 1.0
                st.session_state["pan_x"] = 0.5
                st.session_state["pan_y"] = 0.5
                st.rerun()

    # --- Pan Controls are now integrated into the image canvas ---


def main():
    setup_page()
    setup_styles()
    initialize_session_state()

    # --- AUTO-HEAL: Download weights FIRST if missing ---
    if not os.path.exists(CHECKPOINT_PATH):
        placeholder = st.empty()
        with placeholder.container():
            st.warning("⚠️ Light AI model not found. Downloading automatically... (approx 40MB)")
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                chunk_size = 512 * 1024 
                with open(CHECKPOINT_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = min(1.0, downloaded / total_size)
                                progress_bar.progress(percent)
                            status_text.text(f"📥 Downloaded {downloaded//(1024*1024)}MB...")
                if os.path.getsize(CHECKPOINT_PATH) < 35 * 1024 * 1024:
                     st.error("Download incomplete or corrupt.")
                     os.remove(CHECKPOINT_PATH)
                     st.stop()
                st.success("✅ Model weights verified.")
                time.sleep(1)
                st.cache_resource.clear() # CRITICAL: Clear cache so it tries to load for real
                st.rerun() 
            except Exception as e:
                st.error(f"❌ Failed to download model: {e}")
                if os.path.exists(CHECKPOINT_PATH): os.remove(CHECKPOINT_PATH)
                st.stop()

    # --- STABILITY MIGRATION ---
    if st.session_state.get("image") is not None:
        opts_h, opts_w = st.session_state["image"].shape[:2]
        # HD RESOLUTION UNLOCK (Step Id 1779+)
        # We upped this from 640 -> 1024 to support 360-degree/Panoramic images.
        # Wide images need more pixels to resolve wall edges (straight, right, left, back).
        limit = 800 # Optimized for speed (was 1024) 
        if max(opts_h, opts_w) > limit:
            scale = limit / max(opts_h, opts_w)
            new_w, new_h = int(opts_w * scale), int(opts_h * scale)
            st.session_state["image"] = cv2.resize(st.session_state["image"], (new_w, new_h), interpolation=cv2.INTER_AREA)
            st.session_state["masks"] = [] 
            st.session_state["composited_cache"] = None
            st.session_state["bg_cache"] = None
            sam = get_sam_engine(CHECKPOINT_PATH, MODEL_TYPE)
            if sam: sam.is_image_set = False 
            # REMOVED st.rerun() to prevent sidebar skip bug
            # st.rerun()
    
    # Render Sidebar FIRST (Instant UI)
    device_str = "CUDA" if torch.cuda.is_available() else "CPU"
    
    # We pass None for sam initially to sidebar, or refactor sidebar to not need it for basic UI
    # Let's adjust sidebar to take device string specifically
    
    # Load Model (Session Aware - Optimized for "Instant" feel)
    placeholder = st.empty()
    device_str = "CUDA" if torch.cuda.is_available() else "CPU"
    
    # PERFORMANCE: Only show spinner on first load to avoid "Loading Interruption" on every click
    if not st.session_state.get("engine_ready", False):
        with placeholder.container():
            with st.spinner(f"🚀 Initializing AI Engine on {device_str}..."):
                sam = get_sam_engine(CHECKPOINT_PATH, MODEL_TYPE)
                if sam:
                    st.session_state["engine_ready"] = True
    else:
        sam = get_sam_engine(CHECKPOINT_PATH, MODEL_TYPE)
    
    if not sam:
        st.error("AI Engine could not be initialized. Please check if the model weights are downloaded correctly.")
        st.stop()
        
    placeholder.empty()
    
    # 1. Prepare Base Image (Original + Applied Colors) (Draw Phase)
    # MOVED UP (Step Id 195): Needed for Sidebar Preview
    full_composited = None
    if st.session_state["image"] is not None:
        if st.session_state.get("composited_cache") is None:
             st.session_state["composited_cache"] = composite_image(st.session_state["image"], st.session_state["masks"])
        full_composited = st.session_state["composited_cache"]

    render_sidebar(sam, device_str, full_composited)

    # CENTRALIZED LOADING: Compute embeddings if a new image is present OR if the engine was reset/re-allocated
    if st.session_state["image"] is not None:
        img_id = st.session_state.get("image_path")
        
        # ROBUST LOOP PREVENTION:
        # Only re-analyze if BOTH conditions are true:
        # 1. Session state thinks this is a new image (engine_img_id changed)
        # 2. SAM genuinely doesn't have this image set (is_image_set is False)
        needs_analysis = (
            st.session_state.get("engine_img_id") != img_id and 
            not getattr(sam, 'is_image_set', False)
        )
        
        if needs_analysis:
            
            # --- PROFESSIONAL LOADING OVERLAY (Themed for Dark Mode Studio) ---
            # Inject CSS overlay, replacing standard spinner for a cleaner look
            loading_placeholder = st.empty()
            loading_placeholder.markdown("""
                <style>
                .loader-overlay {
                    position: fixed;
                    top: 0; left: 0; width: 100%; height: 100%;
                    background: rgba(14, 17, 23, 0.95); /* Deep Dark Studio Background */
                    backdrop-filter: blur(12px);
                    z-index: 9999999;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                }
                .loader-spinner {
                    width: 70px;
                    height: 70px;
                    border: 6px solid rgba(255, 255, 255, 0.1);
                    border-top: 6px solid #FF5A5F;
                    border-radius: 50%;
                    animation: spin 1s cubic-bezier(0.68, -0.55, 0.27, 1.55) infinite;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 20px rgba(255, 90, 95, 0.4);
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                .loader-text {
                    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                    font-size: 2.2rem;
                    color: #FFFFFF;
                    font-weight: 700;
                    letter-spacing: -1px;
                }
                .loader-sub {
                    font-family: 'Segoe UI', Roboto, sans-serif;
                    color: #AAAAAA;
                    margin-top: 15px;
                    font-size: 1.2rem;
                    letter-spacing: 0.5px;
                }
                </style>
                <div class="loader-overlay">
                    <div class="loader-spinner"></div>
                    <div class="loader-text">Analyzing Room Structure...</div>
                    <div class="loader-sub">Preparing Studio AI Engine</div>
                </div>
            """, unsafe_allow_html=True)
            
            lock = get_global_lock()
            with lock: 
                try:
                    # RAM PROTECTION: Avoid unnecessary copies
                    img_to_process = st.session_state["image"]
                    
                    # Heavy AI Processing
                    with torch.inference_mode():
                        sam.set_image(img_to_process)
                    
                    # Mark as successfully processed
                    st.session_state["engine_img_id"] = img_id
                    
                    # Cleanup
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    # Clear Loader Immediately
                    loading_placeholder.empty()
                    
                    # Force a refresh to show the result
                    st.rerun()
                    
                except Exception as e:
                    loading_placeholder.empty()
                    st.error(f"Error analyzing image: {e}")
                    if "Out of memory" in str(e):
                        torch.cuda.empty_cache()
                    st.stop()
    
    # Main Workflow
    if st.session_state.get("image") is not None:
        
        # 1. Image Check is already handled in stability migration above
        # Double check validity to prevent crashes
        if st.session_state["image"].size == 0:
            st.error("Loaded image is empty. Please upload a valid image.")
            st.session_state["image"] = None
            st.rerun()

        h, w, c = st.session_state["image"].shape

        h, w, c = st.session_state["image"].shape
        
        # 1. Calc Geometry (Where are we looking?)
        start_x, start_y, view_w, view_h = get_crop_params(
            w, h, 
            st.session_state["zoom_level"], 
            st.session_state["pan_x"], 
            st.session_state["pan_y"]
        )
        
        # 2. Calc Display Geometry
        display_width = 1000 # Reduced safe width
        # Determine crop dimensions
        crop_w = view_w
        crop_h = view_h
        
        # Calculate scale
        scale_factor = display_width / crop_w
        
        # 3. Check for Click in Session State
        # CRITICAL: This key MUST match the rendered component key exactly to capture clicks.
        canvas_key = f"canvas_{st.session_state['render_id']}_{st.session_state['zoom_level']}_{st.session_state['pan_x']:.2f}_{st.session_state['pan_y']:.2f}"
        input_value = st.session_state.get(canvas_key)
        
        if input_value is not None:
            click_x_display, click_y_display = input_value["x"], input_value["y"]
            
            # --- Pan Logic ---
            if st.session_state["zoom_level"] > 1.0:
                # Need to deduce display height for pan zones
                new_h = int(crop_h * scale_factor)
                d_h, d_w = new_h, display_width
                
                margin = 50
                step_size = 0.1 / st.session_state["zoom_level"]
                pan_triggered = False
                
                if click_y_display < margin:
                    st.session_state["pan_y"] = max(0.0, st.session_state["pan_y"] - step_size)
                    pan_triggered = True
                elif click_y_display > d_h - margin:
                    st.session_state["pan_y"] = min(1.0, st.session_state["pan_y"] + step_size)
                    pan_triggered = True
                elif click_x_display < margin:
                    st.session_state["pan_x"] = max(0.0, st.session_state["pan_x"] - step_size)
                    pan_triggered = True
                elif click_x_display > d_w - margin:
                    st.session_state["pan_x"] = min(1.0, st.session_state["pan_x"] + step_size)
                    pan_triggered = True
                    
                if pan_triggered:
                    st.rerun()

            # --- Painting Logic ---
            # Map Display -> Global
            local_x = int(click_x_display / scale_factor)
            local_y = int(click_y_display / scale_factor)
            global_click_x = start_x + local_x
            global_click_y = start_y + local_y
            
            # Clamp
            global_click_x = max(0, min(global_click_x, w - 1))
            global_click_y = max(0, min(global_click_y, h - 1))
            
            click_tuple = (global_click_x, global_click_y)
            last_click = st.session_state.get("last_click_global")
            
            if click_tuple != last_click:
                st.session_state["last_click_global"] = click_tuple
                
                # NEW: SAMPLING MODE Logic
                if st.session_state.get("sampling_mode"):
                    pixel = st.session_state["image"][global_click_y, global_click_x]
                    hex_val = '#%02x%02x%02x' % (pixel[0], pixel[1], pixel[2])
                    st.session_state["picked_sample"] = hex_val
                    st.session_state["sampling_mode"] = False # Reset
                    st.toast(f"Sampled color: {hex_val}")
                    st.rerun()

                # PROCESS CLICK (No Spinner for speed)
                masks_state = st.session_state["masks"]
                active_sel = st.session_state.get("active_selection")
                
                # LOGIC:
                # 1. If Active Selection exists -> Click = Refine this selection
                # 2. If NO Active Selection -> Click = Start NEW selection
                
                if active_sel:
                     # --- REFINE ACTIVE SELECTION (Restored Multi-Point v1.4.3) ---
                     # Added current click to the series
                     active_sel['points'].append(click_tuple)
                     active_sel['labels'].append(st.session_state.get("click_label", 1))
                     
                     # Re-generate mask using ALL points (Contextual Refinement)
                     with st.spinner("AI Refining Selection..."):
                         new_mask = sam.generate_mask(
                             active_sel['points'],
                             active_sel['labels'],
                             level=st.session_state.get("mask_level", None),
                             use_texture_guard=st.session_state.get("use_texture_guard", True)
                         )
                     
                     if new_mask is not None:
                         active_sel['mask'] = new_mask
                         st.toast("✨ Refined selection")
                         
                         st.session_state["composited_cache"] = None 
                         st.session_state["render_id"] += 1
                         st.rerun()
                         
                         # Force redraw
                         st.session_state["composited_cache"] = None 
                         st.session_state["render_id"] += 1
                         st.rerun()

                else:
                     # --- START NEW SELECTION ---
                     # RESILIENCE: Self-healing check
                     if st.session_state.get("engine_img_id") != id(st.session_state["image"]) or not sam.is_image_set:
                         with st.spinner("🔄 AI Re-syncing image..."):
                             sam.set_image(st.session_state["image"])
                             st.session_state["engine_img_id"] = id(st.session_state["image"])

                     mask = sam.generate_mask(
                        [click_tuple],
                        [1], 
                        level=st.session_state.get("mask_level", None)
                     )
                     
                     if mask is not None:
                        # Color & Finish
                        # Use sidebar Color + Finish (if set, else Matte)
                        # We need to capture Finish from UI state or default
                        current_finish = st.session_state.get("active_finish", "Matte") # This might be hidden if no active sel, so default Matte
                        # Actually, if we are in Instant Mode, we should probably expose Finish in the main palette area or Sidebar persistent state.
                        # For now, let's default to Matte or last used?
                        # Let's use "instant_finish" if we can, or just default Matte
                        
                        # Better: Use the Global Palette finish if we add one, but for now specific object finish.
                        # Let's assume Matte for instant, or better, reuse the last used finish.
                        # Or better yet, we can't switch finish easily if the UI is hidden.
                        # Let's assume the user wants the standard finish they picked.
                        
                        paint_color = st.session_state.get("picked_color", "#FFFF00")
                        
                        if st.session_state.get("instant_mode"):
                            # INSTANT PAINT: Skip Active Selection
                            new_layer = {
                                'mask': mask,
                                'color': paint_color,
                                'finish': st.session_state.get("global_finish_pref", "Matte"),
                                'opacity': 0.9, # Match "Apply Format" (Standard Mode uses 0.85-0.9)
                                'visible': True,
                                'name': f"Surface {len(st.session_state['masks'])+1}",
                                'brightness': 0.0, 'contrast': 1.0, 'saturation': 1.0, 'hue': 0.0
                            }
                            st.session_state["masks"].append(new_layer)
                            st.toast(f"⚡ Instant Painted: {paint_color}")
                            
                            st.session_state["composited_cache"] = None 
                            st.session_state["bg_cache"] = None
                            st.session_state["render_id"] += 1
                            st.rerun()
                            
                        else:
                            # STANDARD MODE: Active Selection Preview
                            preview_color = paint_color
                            
                            selection_obj = {
                                'mask': mask,
                                'mask_soft': None, 
                                'sub_masks': [mask],
                                'color': preview_color, # Placeholder
                                'point': click_tuple,
                                'points': [click_tuple],
                                'labels': [1],
                                'visible': True,
                                'name': "Active Selection"
                            }
                            st.session_state["active_selection"] = selection_obj
                            
                            st.session_state["render_id"] += 1
                            st.rerun()


        # 1. Prepare Base Image (Original + Applied Colors)
        # Already calculated above for Sidebar Preview
        full_composited = st.session_state["composited_cache"]

        # --- PREVIEW ACTIVE SELECTION Overlay ---
        # If we have an active selection (workflow step 2), we overlay it on top 
        # of the committed layers so the user can verify it before applying.
        active_sel = st.session_state.get("active_selection")
        if active_sel and active_sel.get('mask') is not None:
             # Create a working copy to avoid mutating the cache
             # PERFORMANCE: This copy is necessary but acceptable for a single active object.
             full_composited = full_composited.copy()
             
             sel_mask = active_sel['mask']
             if sel_mask.shape[:2] == full_composited.shape[:2]:
                  # Prepare Overlay Color (e.g. User selected color or High-Contrast Default)
                  # We use the 'color' from the object (which syncs with the picker)
                  hex_color = active_sel.get('color', '#FFFF00')
                  
                  # Convert hex to RGB tuple
                  hex_color = hex_color.lstrip('#')
                  rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                  
                  # Apply Real Paint Simulation for Preview
                  # This ensures the user sees exactly what they will get (Texture + LAB color)
                  blend_opacity = active_sel.get('opacity', 0.85)
                  
                  # Use the engine's apply_color for correct texture preservation
                  preview_layer = ColorEngine.apply_material(
                      full_composited, 
                      sel_mask, 
                      active_sel.get('color', '#FFFF00'), 
                      finish=st.session_state.get("active_finish", "Matte"), # Live preview of finish
                      intensity=blend_opacity
                  )
                  
                  # Update the composite
                  full_composited = preview_layer
                  
                  # Add Outline (Stroke) for precision verification
                  # Dilate slightly to create border? Or just findContours.
                  # Contours is cleaner.
                  mask_uint8 = sel_mask.astype(np.uint8)
                  contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                  # Draw thin white outline for visibility against dark colors
                  cv2.drawContours(full_composited, contours, -1, (255, 255, 255), 1)
        
        if st.session_state.get("show_comparison", False):
            # Render Comparison Slider (Read Only)
            st.info("👀 Comparison Mode Active. Toggle off in sidebar to edit.")
            
            # Ensure images are same size/type
            img1 = st.session_state["image"] # Original
            img2 = full_composited          # Edited
            
            image_comparison(
                img1=Image.fromarray(img1),
                img2=Image.fromarray(img2),
                label1="Original",
                label2="Your Design",
                width=1000, # Match display width for consistency
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True
            )
            return # Stop execution here for view mode

        # 2. Apply Zoom/Crop
        h, w, c = full_composited.shape
        start_x, start_y, view_w, view_h = get_crop_params(
            w, h, 
            st.session_state["zoom_level"], 
            st.session_state["pan_x"], 
            st.session_state["pan_y"]
        )
        
        cropped_view = full_composited[start_y:start_y+view_h, start_x:start_x+view_w]
        
        # 3. Interactive Display
        # Resize for display consistency
        display_width = 1000
        crop_h, crop_w, _ = cropped_view.shape
        
        # Calculate scale to fit display layout, handling both upscale (zoom) and downscale
        
        # PERFORMANCE: Cache Display Image to prevent unnecessary re-processing/flicker
        current_disp_params = (
            st.session_state["render_id"], 
            st.session_state["zoom_level"], 
            st.session_state["pan_x"], 
            st.session_state["pan_y"]
        )
        
        if st.session_state.get("last_disp_params") != current_disp_params:
            # Recompute Display Image
            scale_factor = display_width / crop_w
            new_h = int(crop_h * scale_factor)
            
            display_image = cv2.resize(cropped_view, (display_width, new_h), interpolation=cv2.INTER_LINEAR)
            
            if st.session_state["zoom_level"] > 1.0:
                display_image = overlay_pan_controls(display_image)
            
            st.session_state["cached_display_image"] = display_image
            st.session_state["last_disp_params"] = current_disp_params
            st.session_state["last_scale_factor"] = scale_factor
        
        final_display_image = st.session_state.get("cached_display_image")
        scale_factor = st.session_state.get("last_scale_factor", 1.0)

        if final_display_image is None or final_display_image.size == 0:
            st.warning("⚠️ Rendering issue: Image preview is empty. Try refreshing the page.")
            return

        # Container for the image
        with st.container():
            # Ensure explicit strict types for the component
            final_display_image = np.ascontiguousarray(final_display_image, dtype=np.uint8)
            
            # Dynamic key ensures component resets on view change, preventing infinite pan loops
            # PERFORMANCE: We must use 'render_id' in the key to force the component to reset.
            # Otherwise, clicking the same pixel twice (e.g. to re-paint) won't register.
            canvas_key = f"canvas_{st.session_state['render_id']}_{st.session_state['zoom_level']}_{st.session_state['pan_x']:.2f}_{st.session_state['pan_y']:.2f}"
            
            # OPTIMIZATION: Convert to JPEG Base64 to speed up the "flash" reload
            # PNG (default) is too slow and causes visible scanlines. JPEG is instant.
            try:
                value = streamlit_image_coordinates(
                    Image.fromarray(final_display_image),
                    key=canvas_key,
                    width=display_width 
                )
            except Exception as e:
                st.error(f"Display Error: {e}")
                value = None

        # 4. Zoom Controls
        render_zoom_controls()

        # 5. ACTION BAR (New Centralized Control v2.4)
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        active_sel = st.session_state.get("active_selection")
        
        with col1:
            if st.button("⏪ Undo Last", help="Remove the last painted surface", use_container_width=True):
                if st.session_state["masks"]:
                    st.session_state["masks"].pop()
                    st.session_state["composited_cache"] = None 
                    st.session_state["bg_cache"] = None
                    st.session_state["render_id"] += 1
                    st.rerun()
        
        with col2:
            if active_sel:
                if st.button("❌ Discard", help="Discard current selection", use_container_width=True):
                    st.session_state["active_selection"] = None
                    st.session_state["render_id"] += 1
                    st.rerun()
            else:
                if st.button("🧹 Clear All", help="Clear all paint layers", use_container_width=True):
                    st.session_state["masks"] = []
                    st.session_state["composited_cache"] = None 
                    st.session_state["bg_cache"] = None
                    st.session_state["render_id"] += 1
                    st.rerun()

        with col3:
             # Comparison Toggle in the main bar for convenience
             show_comp = st.checkbox("◑ Compare Original", key="compare_toggle", help="Toggle to see original image")
             st.session_state["show_comparison"] = show_comp

        with col4:
            if active_sel:
                if st.button("✨ Apply Surface", type="primary", use_container_width=True):
                    # Commit Active Selection to Masks
                    new_layer = active_sel
                    new_layer['name'] = f"Surface {len(st.session_state['masks'])+1}"
                    new_layer['visible'] = True
                    # Sync color and finish
                    new_layer['color'] = st.session_state.get("picked_color", "#FFFFFF")
                    new_layer['finish'] = st.session_state.get("active_finish", "Matte")
                    new_layer.update({
                        'brightness': 0.0, 'contrast': 1.0, 'saturation': 1.0, 
                        'hue': 0.0, 'opacity': 1.0
                    })
                    st.session_state["masks"].append(new_layer)
                    # Cleanup
                    st.session_state["active_selection"] = None
                    st.session_state["composited_cache"] = None 
                    st.session_state["bg_cache"] = None
                    st.session_state["render_id"] += 1
                    st.rerun()

        # 6. Handle Click Event
        pass

    else:
        # Landing Page (Asian Paints Style)
        st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 70vh; text-align: center;">
                <div style="background: #1A1A1A; padding: 3rem; border-radius: 24px; box-shadow: 0 20px 50px rgba(0,0,0,0.3); max-width: 600px; border: 1px solid #333;">
                    <h1 style="font-size: 3.5rem; margin-bottom: 1rem; background: linear-gradient(135deg, #FF5A5F, #FF8084); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Visualizer Studio</h1>
                    <p style="font-size: 1.2rem; color: #AAA; margin-bottom: 2rem;">Transform your space with AI-powered precision. Choose a room photo to begin your color journey.</p>
                    <div style="padding: 2rem; border: 2px dashed #444; border-radius: 16px; background: #222;">
                        <p style="color: #CCC; margin-bottom: 1rem;">Click 'Browse' in the sidebar or drag an image here</p>
                        <p style="font-size: 0.9rem; color: #888;">Supported: JPG, PNG • Max 200MB</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
