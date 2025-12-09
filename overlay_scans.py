import numpy as np
from PIL import Image
import time
from tqdm import tqdm
import sys

from skimage.registration import phase_cross_correlation
from skimage.transform import rotate
from skimage.filters import sobel
from scipy.ndimage import shift as nd_shift

import tkinter as tk
from tkinter import filedialog, messagebox
from functools import partial

from PIL import ImageTk
import openslide

def load_preview(path):
    if path.lower().endswith(".svs"):
        slide = openslide.OpenSlide(path)
        thumb = slide.get_thumbnail((300, 300))
        return thumb
    else:
        img = Image.open(path)
        img.thumbnail((300, 300))  # Fit preview
        return img

def update_preview(widget, path):
    if not path:
        return
    img = load_preview(path)
    tk_img = ImageTk.PhotoImage(img)
    widget.image = tk_img  # prevent garbage collection
    widget.config(image=tk_img)

def load_image_any(path):
    """Loads TIFF/PNG/JPG normally, loads SVS with OpenSlide at full resolution."""

    if path.lower().endswith(".svs"):
        slide = openslide.OpenSlide(path)
        # Level 0 = highest resolution
        w, h = slide.dimensions
        region = slide.read_region((0, 0), 0, (w, h))  # RGBA PIL Image
        img = region.convert("RGB")  # strip alpha
        return np.array(img)

    # Otherwise normal image
    return np.array(Image.open(path))

def create_overlay(
        backlit_filepath,
        fibi_filepath,
        output_filepath = "output.tiff",
        downsample_factor = 2,
        upsample_factor = 10,
        fake_mode=False,
        progress_callback=None,
        backlit_opacity=0.25):
    if fake_mode:
        print(backlit_opacity)
        for i in tqdm(range(14), desc="Progress [TESTING]"):
            time.sleep(0.05)
            if progress_callback:
                progress_callback((i+1)/14*100)
        return
    # --- Load images ---
    auto_rgb = load_image_any(fibi_filepath)
    back_rgb = load_image_any(backlit_filepath)

    if progress_callback:
        progress_callback(5)  # 5% done after loading images

    def to_gray(arr):
        if arr.ndim == 3:
            return arr.mean(axis=2).astype(np.float32)
        return arr.astype(np.float32)

    auto_gray = to_gray(auto_rgb)
    back_gray = to_gray(back_rgb)

    # Crop to common size
    h = min(auto_gray.shape[0], back_gray.shape[0])
    w = min(auto_gray.shape[1], back_gray.shape[1])
    auto_gray = auto_gray[:h, :w]
    back_gray = back_gray[:h, :w]       
    auto_rgb = auto_rgb[:h, :w]
    back_rgb = back_rgb[:h, :w]

    # --- Feature preprocessing: Sobel gradient magnitude ---
    auto_feat = sobel(auto_gray)
    back_feat = sobel(back_gray)

    # Downsample to speed up MI computations
    ds = downsample_factor
    auto_feat_ds = auto_feat[::ds, ::ds]
    back_feat_ds = back_feat[::ds, ::ds]

    H_ds, W_ds = auto_feat_ds.shape

    # --- Translation baseline from phase_cross_correlation (on downsampled features) ---
    shift_est_ds, error, _ = phase_cross_correlation(
        back_feat_ds, auto_feat_ds, upsample_factor=upsample_factor
    )
    print("Baseline shift (downsampled) dy,dx:", shift_est_ds, "error:", error)

    # Convert shift to full-res pixels (since ds=2, multiply by ds)
    shift_est_full = shift_est_ds * ds
    print("Baseline shift (full-res approx) dy,dx:", shift_est_full)

    if progress_callback:
        progress_callback(10)  # 10% after baseline

    # For MI search we'll work in downsampled coordinates; keep ds-space shift
    base_shift = shift_est_ds

    # --- Mutual information utility ---
    def mutual_information(a, b, bins=64):
        # Flatten and compute 2D histogram
        a = a.ravel()
        b = b.ravel()
        hgram, _, _ = np.histogram2d(a, b, bins=bins)
        # Convert to joint probability
        pxy = hgram / np.sum(hgram)
        px = np.sum(pxy, axis=1)  # marginal for a
        py = np.sum(pxy, axis=0)  # marginal for b
        px_py = px[:, None] * py[None, :]
        nz = pxy > 0
        mi = np.sum(pxy[nz] * np.log(pxy[nz] / (px_py[nz] + 1e-12)))
        return mi

    # --- Search over small rotations + residual translations around base shift ---
    angle_range = (-.5, .5)
    angle_step = 0.25
    delta_range = np.arange(-1, 1, 0.5)  # residual shift in ds pixels around base_shift
    best = None  # (mi, angle, dy_total, dx_total)
    angles = np.arange(angle_range[0], angle_range[1] + 1e-9, angle_step)

    # We'll crop to central region to avoid edge artifacts after shifting
    crop_margin = 300  # in downsampled pixels
    def central_crop(img):
        return img[crop_margin:-crop_margin, crop_margin:-crop_margin]

    back_c = central_crop(back_feat_ds)

    total_steps = len(angles) * len(delta_range) ** 2
    step_count = 0

    for ang in angles:
        # Rotate auto feature (downsampled)
        auto_rot = rotate(
            auto_feat_ds,
            angle=ang,
            resize=False,
            order=1,
            mode="constant",
            cval=0,
            preserve_range=True
        )
        for dy_res in delta_range:
            for dx_res in delta_range:
                step_count += 1
                dy_total = base_shift[0] + dy_res
                dx_total = base_shift[1] + dx_res

                auto_shift = nd_shift(
                    auto_rot,
                    shift=(dy_total, dx_total),
                    order=1,
                    mode="constant",
                    cval=0
                )
                auto_c = central_crop(auto_shift)

                # Ensure same shape
                if auto_c.shape != back_c.shape:
                    continue

                mi = mutual_information(back_c, auto_c, bins=64)

                if best is None or mi > best[0]:
                    best = (mi, ang, dy_total, dx_total)

                # --- update progress ---
                if progress_callback:
                    progress_callback(10 + 70*step_count/total_steps)  # 10-80%

    best_mi, best_ang, best_dy_ds, best_dx_ds = best

    # Convert best shift back to full-res pixels
    best_shift_full = np.array([best_dy_ds, best_dx_ds]) * ds
    print("dx:", best_shift_full[0], "dy:", best_shift_full[1], "dtheta:", best_ang)

    if progress_callback:
        progress_callback(85)  # MI search done

    # --- Apply best transform to full-res RGB auto image ---
    # Rotate full-res image
    auto_rot_full = rotate(
        auto_rgb,
        angle=best_ang,
        resize=False,
        order=1,
        mode="constant",
        cval=0,
        preserve_range=True
    ).astype(np.float32)

    auto_aligned_full = np.zeros_like(auto_rot_full)
    for c in range(auto_rot_full.shape[2]):
        auto_aligned_full[..., c] = nd_shift(
            auto_rot_full[..., c],
            shift=best_shift_full,
            order=1,
            mode="constant",
            cval=0
        )

    # Normalize for display
    def norm01(arr):
        arr = arr.astype(np.float32)
        mn, mx = arr.min(), arr.max()
        if mx <= mn:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    back_n = norm01(back_rgb)
    auto_aligned_n = norm01(auto_aligned_full)
    overlay = backlit_opacity * back_n + (1-backlit_opacity) * auto_aligned_n

    # ---- SAVE FINAL TIFF ----
    # Convert overlay_blend back to 0-255 uint8 for saving
    overlay_uint8 = (overlay * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay_uint8).save(output_filepath, compression="tiff_lzw")
    print("Wrote final result to:", output_filepath)
    if progress_callback:
        progress_callback(100)  # Done

def main():
    # Store file paths
    backlit_path = None
    fibi_path = None
    output_path = None

    def pick_backlit():
        nonlocal backlit_path
        backlit_path = filedialog.askopenfilename(
            title="Select Backlit Image",
            filetypes=[("Image Files", ".tiff .tif .svs")]
        )
        back_button.config(text=f"Backlit: {backlit_path.split('/')[-1]}" if backlit_path else "Select Backlit Image")
        root.lift()
        root.focus_force()
        update_preview(preview_canvas1, backlit_path)


    def pick_fibi():
        nonlocal fibi_path
        fibi_path = filedialog.askopenfilename(
            title="Select FIBI Image",
            filetypes=[("Image Files", ".tiff .tif .svs")]
        )
        fibi_button.config(text=f"FIBI: {fibi_path.split('/')[-1]}" if fibi_path else "Select FIBI Image")
        root.lift()
        root.focus_force()
        update_preview(preview_canvas2, fibi_path)


    def pick_output():
        nonlocal output_path
        output_path = filedialog.asksaveasfilename(
            title="Save Overlay As",
            defaultextension=".tiff",
            filetypes=[("TIFF", ".tiff .tif")]
        )
        output_button.config(text=f"Output: {output_path.split('/')[-1]}" if output_path else "Select Output File")
        root.lift()
        root.focus_force()

    import threading

    def run_overlay_threaded():
        def worker():
            try:
                if backlit_path and fibi_path:
                    create_overlay(
                        backlit_path,
                        fibi_path,
                        output_path,
                        fake_mode=False,
                        progress_callback=lambda p: progress_var.set(p),
                        backlit_opacity=my_slider.get()
                    )
                else:
                    create_overlay(
                        backlit_path,
                        fibi_path,
                        output_path,
                        fake_mode=True,
                        progress_callback=lambda p: progress_var.set(p),
                        backlit_opacity=my_slider.get()
                    )
                progress_var.set(100)
                messagebox.showinfo("Success", "Overlay created successfully!")
            except Exception as e:
                progress_var.set(0)
                messagebox.showerror("Error", str(e))

        threading.Thread(target=worker, daemon=True).start()

    from tkinter import ttk

    # ===== Tkinter GUI =====
    root = tk.Tk()
    root.title("FIBI Overlay Tool")
    root.geometry("700x300")  # Wider window
    BG = "#F0F0F0"

    # ---- ttk Styling ----
    style = ttk.Style()
    style.theme_use("clam")
    root.configure(background=BG)
    style.configure("TFrame", background=BG)
    style.configure("TLabel", font=("Helvetica", 11), background=BG)
    style.configure("Title.TLabel", font=("Helvetica", 15, "bold"), background=BG)
    style.configure("TButton", font=("Helvetica", 10), padding=4)

    # ====== LAYOUT ======
    main_frame = ttk.Frame(root)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # --- LEFT PANEL (Controls) ---
    left = ttk.Frame(main_frame)
    left.pack(side="left", fill="y", padx=10)

    title = ttk.Label(left, text="FIBI Overlay Manager", style="Title.TLabel")
    title.pack(pady=10)

    back_button = ttk.Button(left, text="Select Backlit Image", command=pick_backlit, width=25)
    back_button.pack(pady=5)

    fibi_button = ttk.Button(left, text="Select FIBI Image", command=pick_fibi, width=25)
    fibi_button.pack(pady=5)

    output_button = ttk.Button(left, text="Select Output File", command=pick_output, width=25)
    output_button.pack(pady=5)

    def update_label(event=None):
        value = slider.get()
        label.config(text=f"Backlit Opacity: {value:.1f}%")  # ← 2 decimal places
    
    value = tk.DoubleVar()

    slider = ttk.Scale(left, from_=0, to=100, length=200, variable=value, command=update_label)
    slider.pack(pady=5)

    label = ttk.Label(left, text="Backlit Opacity: 0.0")
    label.pack()

    run_btn = ttk.Button(left, text="Create Overlay", command=run_overlay_threaded, width=25)
    run_btn.pack(pady=10)

    progress_var = tk.DoubleVar()
    progress_label = ttk.Label(left, text=f"Progress:")
    progress_label.pack(pady=(10,2))

    progress_bar = ttk.Progressbar(left, maximum=100, length=200, variable=progress_var)
    progress_bar.pack(pady=2)


    # --- RIGHT PANEL (Two Side-by-Side Previews) ---
    right = ttk.Frame(main_frame)
    right.pack(side="right", fill="both", expand=True, padx=10)

    # Subframe for row layout
    preview_row = ttk.Frame(right)
    preview_row.pack(fill="both", expand=True)

    # --- Backlit Column ---
    col1 = ttk.Frame(preview_row, width=200, height=200)
    col1.pack(side="left", fill="both", expand=False, padx=(0,5))
    col1.pack_propagate(False)  # Prevent frame from shrinking

    preview_label1 = ttk.Label(col1, text="Backlit", anchor="center")
    preview_label1.pack()
    preview_canvas1 = tk.Label(col1, bg="black", width=140, height=180)
    preview_canvas1.pack(pady=5)

    # --- FIBI Column ---
    col2 = ttk.Frame(preview_row, width=200, height=200)
    col2.pack(side="left", fill="both", expand=False, padx=(5,0))
    col2.pack_propagate(False)

    preview_label2 = ttk.Label(col2, text="FIBI", anchor="center")
    preview_label2.pack()
    preview_canvas2 = tk.Label(col2, bg="black", width=140, height=180)
    preview_canvas2.pack(pady=5)

    # Keep window focused after dialogs
    root.lift()
    root.focus_force()
    root.mainloop()

if __name__ == "__main__":
    if hasattr(sys, 'frozen'):
        import multiprocessing
        multiprocessing.freeze_support()
    main()
