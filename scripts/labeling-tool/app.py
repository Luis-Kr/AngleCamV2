import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageOps
import csv
import os
import json
import shutil

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.stats as stats
import numpy as np
from scipy.optimize import minimize
import pandas as pd
from pathlib import Path

# For Pillow >=10.0.0, use Image.Resampling.LANCZOS; otherwise, fallback to Image.LANCZOS.
try:
    resample_filter = Image.Resampling.LANCZOS
except AttributeError:
    resample_filter = Image.LANCZOS

class BetaDistributionFitter:
    def __init__(self, 
                 sd_fraction=0.2,
                 sim_per_pic=50,
                 angle_resolution=43,
                 scaling_factor=10,
                 ncp=0):
        self.sd_fraction = sd_fraction
        self.sim_per_pic = sim_per_pic
        self.angle_resolution = angle_resolution
        self.scaling_factor = scaling_factor
        self.ncp = ncp
        self.bins = np.linspace(0, 90, angle_resolution)

    def fit_mle(self, data):
        """Fit beta distribution using Maximum Likelihood Estimation"""
        # Scale angles from 0-90 to 0-1
        scaled_data = np.clip(data / 90.0, 0.00001, 0.99999)
        
        # Fit beta distribution
        a, b, loc, scale = stats.beta.fit(scaled_data, floc=0, fscale=1)
        
        # Calculate standard errors through Fisher Information Matrix
        x = scaled_data
        def neg_log_likelihood(params):
            return -np.sum(stats.beta.logpdf(x, params[0], params[1]))
        
        result = minimize(neg_log_likelihood, [a, b], method='Nelder-Mead')
        hess_inv = result.hess_inv if hasattr(result, 'hess_inv') else np.eye(2)
        std_errors = np.sqrt(np.diag(hess_inv))
        
        return {'estimate': (a, b), 'sd': std_errors}

    def generate_distributions(self, fit_result):
        """Generate reference and simulated distributions"""
        x = np.linspace(0, 1, self.angle_resolution)
        
        # Incorporate self.ncp by shifting the x values.
        x_adj = np.clip(x - self.ncp/90.0, 0.00001, 0.99999)
        
        # Generate reference distribution using the shifted x values
        ref_dist = stats.beta.pdf(x_adj, fit_result['estimate'][0], fit_result['estimate'][1])
        
        # Handle invalid values and normalize to sum to 1
        ref_dist = np.nan_to_num(ref_dist, nan=0.0, posinf=0.0, neginf=0.0)
        sum_ref = np.sum(ref_dist)
        ref_dist = ref_dist / sum_ref if sum_ref > 0 else ref_dist
        
        # Generate simulated distributions
        sims = []
        for _ in range(self.sim_per_pic):
            # Sample parameters with error
            a = fit_result['estimate'][0] + np.random.normal(0, fit_result['sd'][0]) * self.sd_fraction
            b = fit_result['estimate'][1] + np.random.normal(0, fit_result['sd'][1]) * self.sd_fraction
            
            try:
                # Use the same shifted x values for simulation
                dist = stats.beta.pdf(x_adj, a, b)
                
                # Handle invalid values and normalize to sum to 1
                dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
                sum_dist = np.sum(dist)
                if sum_dist > 0:
                    dist = dist / sum_dist
                    sims.append(dist)
            except:
                continue
        
        return ref_dist, np.array(sims)

    @staticmethod
    def mean_angle(pred: np.ndarray) -> float:
        """Calculate mean angle from prediction array."""
        angle_res = pred.shape[0]
        angles = np.linspace(0, 90, angle_res)
        return np.sum(pred * angles)

    def create_simulation_plot(self, angles, fit_result, output_path):
        """Create plot with histogram, fitted distribution, and simulation lines"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram of actual data
        counts, bins, _ = ax.hist(angles, bins=self.bins, density=True, 
                                 alpha=0.6, label='Data', color='lightblue')
        
        # Plot fitted beta distribution (reference)
        x = np.linspace(0, 90, 200)
        x_adj = np.clip((x - self.ncp)/90.0, 0.00001, 0.99999)
        a, b = fit_result['estimate']
        y = stats.beta.pdf(x_adj, a, b) / 90.0
        
        # Calculate mean angle for fitted distribution
        y_for_mean = stats.beta.pdf(np.linspace(0, 1, self.angle_resolution) - self.ncp/90.0, a, b)
        y_for_mean = np.nan_to_num(y_for_mean, nan=0.0, posinf=0.0, neginf=0.0)
        sum_y = np.sum(y_for_mean)
        if sum_y > 0:
            y_for_mean = y_for_mean / sum_y
        mean_angle = self.mean_angle(y_for_mean)
        
        ax.plot(x, y, 'r-', lw=3, label=f'Fitted Beta (mean: {mean_angle:.1f}°)')
        
        # Plot simulated distributions as thin lines
        colors = plt.cm.rainbow(np.linspace(0, 1, self.sim_per_pic))
        for i in range(self.sim_per_pic):
            a_sim = fit_result['estimate'][0] + np.random.normal(0, fit_result['sd'][0]) * self.sd_fraction
            b_sim = fit_result['estimate'][1] + np.random.normal(0, fit_result['sd'][1]) * self.sd_fraction
            try:
                y_sim = stats.beta.pdf(x_adj, a_sim, b_sim) / 90.0
                ax.plot(x, y_sim, '-', lw=0.5, alpha=0.3, color=colors[i])
            except:
                continue
        
        # Add a legend entry for simulations
        ax.plot([], [], '-', lw=0.5, alpha=0.3, color='gray', label=f'{self.sim_per_pic} Simulations')
        
        ax.set_xlabel('Inclination Angle (degrees)')
        ax.set_ylabel('Density')
        ax.set_title(f'Leaf Inclination Angle Distribution with Beta Fit and Simulations\nMean Angle: {mean_angle:.1f}°')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = str(Path(output_path).with_suffix('')) + '.png'
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return plot_path

    def process_angles(self, angles, output_path):
        """Process angles and generate simulations"""
        if len(angles) < 3:
            return None, None
            
        try:
            # Create labels directory
            output_dir = Path(output_path).parent
            labels_dir = output_dir / "labels"
            labels_dir.mkdir(exist_ok=True)
            
            # Get base filename
            base_name = Path(output_path).stem
            
            # Fit distribution and generate variants
            fit_result = self.fit_mle(angles)
            ref_dist, sims = self.generate_distributions(fit_result)
            
            # Create plot in labels directory
            plot_path = labels_dir / f"{base_name}_simulation_plot.png"
            self.create_simulation_plot(angles, fit_result, str(plot_path))
            
            # Combine reference and simulations for CSV output
            if len(sims) > 0:
                # Remove first and last points and combine
                all_dist = np.vstack([ref_dist[1:-1], sims[:, 1:-1]])
            else:
                all_dist = ref_dist[1:-1].reshape(1, -1)
            
            # Save simulation results in labels directory
            sim_path = labels_dir / f"{base_name}_sim.csv"
            np.savetxt(sim_path, all_dist, delimiter=',', fmt='%.6f')
            
            return str(sim_path), str(plot_path)
            
        except Exception as e:
            print(f"Error in simulation: {e}")
            return None, None

class LeafAngleLabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Leaf Angle Calculator")
        
        # Image / Zoom Variables
        self.image = None           
        self.original_image = None  
        self.image_tk = None        
        self.image_path = None      
        self.zoom = 1.0             
        
        # Point Variables
        self.current_points = []     
        self.point_ids = []         
        
        self.results = []           
        self.highlighted_point_ids = []
        
        # Distribution Plot Variables
        self.distribution_canvas = None  
        self.distribution_image = None   
        self.grid_ids = []               
        
        # Project Mode Variables
        self.project_data = None
        self.project_name = None     
        self.current_filename = None  
        
        # Store the last used directory for opening/creating project files.
        self.last_project_dir = os.getcwd()
        
        # Initialize beta distribution fitter
        self.beta_fitter = BetaDistributionFitter(
            sd_fraction=0.2,
            sim_per_pic=50,
            angle_resolution=43,
            scaling_factor=10,
            ncp=0
        )
        
        self.setup_ui()
        self.configure_styles()
    
    def configure_styles(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('TButton', font=('Helvetica', 11), padding=6, relief='flat')
        style.map('TButton', background=[('active', '#347083')])
        style.configure('TLabel', font=('Helvetica', 11), foreground='#333333')
        style.configure('Treeview', font=('Helvetica', 10), rowheight=30, fieldbackground='white')
        style.configure('Treeview.Heading', font=('Helvetica', 11, 'bold'),
                        foreground='white', background='#3A3A3A')
    
    def setup_ui(self):
        # Set minimum window size and make it resizable
        self.root.minsize(800, 600)
        self.root.geometry("1200x800")  # Set initial size
        
        # Create a PanedWindow: left for image canvas; right for controls
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left frame: for image canvas
        self.left_frame = ttk.Frame(paned)
        paned.add(self.left_frame, weight=3)
        
        # Right frame: for controls and project toolbar
        self.right_frame = ttk.Frame(paned)
        paned.add(self.right_frame, weight=1)
        
        # Left: Canvas with scrollbars for large images
        canvas_frame = ttk.Frame(self.left_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        
        # Create canvas
        self.canvas = tk.Canvas(canvas_frame, bg='grey', 
                               yscrollcommand=v_scrollbar.set,
                               xscrollcommand=h_scrollbar.set)
        
        # Configure scrollbars
        v_scrollbar.config(command=self.canvas.yview)
        h_scrollbar.config(command=self.canvas.xview)
        
        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        # Mouse wheel zoom support
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)    # Linux scroll down
        
        # Right: Project Toolbar - organized in rows for better responsive layout
        self.project_toolbar = ttk.Frame(self.right_frame)
        self.project_toolbar.pack(padx=5, pady=5, fill=tk.X)
        
        # Row 1: Project management
        toolbar_row1 = ttk.Frame(self.project_toolbar)
        toolbar_row1.pack(fill=tk.X, pady=2)
        self.new_project_button = ttk.Button(toolbar_row1, text="New Project", command=self.new_project)
        self.new_project_button.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        self.open_project_button = ttk.Button(toolbar_row1, text="Open Project", command=self.open_project)
        self.open_project_button.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        self.add_image_button = ttk.Button(toolbar_row1, text="Add Image", command=self.add_image)
        self.add_image_button.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        
        # Row 2: Navigation
        toolbar_row2 = ttk.Frame(self.project_toolbar)
        toolbar_row2.pack(fill=tk.X, pady=2)
        self.prev_button = ttk.Button(toolbar_row2, text="◀ Prev", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        self.next_button = ttk.Button(toolbar_row2, text="Next ▶", command=self.next_image)
        self.next_button.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        
        # Row 3: Zoom controls
        toolbar_row3 = ttk.Frame(self.project_toolbar)
        toolbar_row3.pack(fill=tk.X, pady=2)
        self.zoom_in_button = ttk.Button(toolbar_row3, text="Zoom In", command=self.zoom_in)
        self.zoom_in_button.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        self.zoom_out_button = ttk.Button(toolbar_row3, text="Zoom Out", command=self.zoom_out)
        self.zoom_out_button.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        self.fit_button = ttk.Button(toolbar_row3, text="Fit to Window", command=self.fit_to_window)
        self.fit_button.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        
        # Project navigation label
        self.project_nav_label = ttk.Label(self.project_toolbar, text="No project loaded")
        self.project_nav_label.pack(pady=2)
        
        # Right: Info Frame (Below Toolbar)
        self.info_frame = ttk.Frame(self.right_frame)
        self.info_frame.pack(padx=5, pady=5, fill=tk.X)
        self.project_info_label = ttk.Label(self.info_frame, text="Project: None")
        self.project_info_label.pack(anchor="w")
        self.filename_info_label = ttk.Label(self.info_frame, text="Filename: None")
        self.filename_info_label.pack(anchor="w")
        
        
        # Right: Processing Buttons
        self.save_button = ttk.Button(self.right_frame, text="Save CSV", command=self.save_csv)
        self.save_button.pack(pady=5, fill=tk.X, padx=5)
        self.dist_button = ttk.Button(self.right_frame, text="Calculate Distribution", command=self.plot_distribution)
        self.dist_button.pack(pady=5, fill=tk.X, padx=5)
        
        # Right: Treeview for Leaf Results
        self.results_label = ttk.Label(self.right_frame, text="Leaf Results:")
        self.results_label.pack(pady=5)
        columns = ("id", "x", "Y", "angle", "rolling")
        self.results_tree = ttk.Treeview(self.right_frame, columns=columns, show="headings", selectmode="browse")
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=60)
        self.results_tree.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.results_tree.bind("<Double-1>", self.on_treeview_double_click)
        self.results_tree.bind("<<TreeviewSelect>>", self.on_treeview_select)
        self.delete_leaf_button = ttk.Button(self.right_frame, text="Delete Selected Leaf", command=self.delete_selected_leaf)
        self.delete_leaf_button.pack(pady=5, fill=tk.X, padx=5)
        
        # Right: Distribution Plot Frame
        self.plot_frame = ttk.Frame(self.right_frame)
        self.plot_frame.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
    
    # ------------- Zoom & Grid Functions -------------
    def update_canvas_image(self):
        if not self.original_image:
            return
        new_width = int(self.original_image.width * self.zoom)
        new_height = int(self.original_image.height * self.zoom)
        self.image = self.original_image.resize((new_width, new_height), resample=resample_filter)
        self.image_tk = ImageTk.PhotoImage(self.image)
        
        # Clear canvas and create image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk, tags="image")
        
        # Update scroll region to match image size
        self.canvas.configure(scrollregion=(0, 0, new_width, new_height))
        
        self.draw_grid()
        # Re-draw saved points
        for res in self.results:
            if "point" in res:
                x, y = res["point"]
                self.draw_point(x * self.zoom, y * self.zoom, res["id"])
    
    def draw_grid(self):
        for gid in self.grid_ids:
            self.canvas.delete(gid)
        self.grid_ids = []
        if not self.original_image:
            return
        orig_width = self.original_image.width
        orig_height = self.original_image.height
        
        # Draw 20 evenly spaced crosses (4x5 grid)
        cols, rows = 5, 4
        for i in range(cols):
            for j in range(rows):
                x = (i + 0.5) * (orig_width / cols) * self.zoom
                y = (j + 0.5) * (orig_height / rows) * self.zoom
                # Draw cross (+ shape) - scale with zoom but keep readable
                cross_size = max(3, min(10, 5 / self.zoom))
                # Horizontal line
                gid1 = self.canvas.create_line(x - cross_size, y, x + cross_size, y, fill="red", width=2)
                # Vertical line
                gid2 = self.canvas.create_line(x, y - cross_size, x, y + cross_size, fill="red", width=2)
                self.grid_ids.extend([gid1, gid2])
    
    def zoom_in(self):
        self.zoom *= 1.2
        self.update_canvas_image()
    
    def zoom_out(self):
        self.zoom /= 1.2
        self.update_canvas_image()
    
    def fit_to_window(self):
        """Fit image to current canvas size"""
        if not self.original_image:
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
            
        # Calculate zoom to fit image in canvas
        zoom_x = canvas_width / self.original_image.width
        zoom_y = canvas_height / self.original_image.height
        self.zoom = min(zoom_x, zoom_y) * 0.95  # 95% to leave some margin
        
        self.update_canvas_image()
    
    def on_canvas_configure(self, event):
        """Handle canvas resize events"""
        if hasattr(self, 'original_image') and self.original_image:
            # Update scroll region
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel zoom"""
        if not self.original_image:
            return
        
        # Get mouse position relative to canvas
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Determine zoom direction
        if event.delta > 0 or event.num == 4:  # Zoom in
            zoom_factor = 1.1
        else:  # Zoom out
            zoom_factor = 0.9
        
        # Apply zoom
        old_zoom = self.zoom
        self.zoom *= zoom_factor
        
        # Keep zoom within reasonable bounds
        self.zoom = max(0.1, min(self.zoom, 10.0))
        
        # Update image
        self.update_canvas_image()
        
        # Adjust scroll position to zoom towards mouse cursor
        if old_zoom != self.zoom:
            zoom_ratio = self.zoom / old_zoom
            new_canvas_x = canvas_x * zoom_ratio
            new_canvas_y = canvas_y * zoom_ratio
            
            # Calculate how much to scroll to keep mouse position centered
            dx = new_canvas_x - canvas_x
            dy = new_canvas_y - canvas_y
            
            # Get current scroll position
            scrollregion = self.canvas.cget("scrollregion").split()
            if len(scrollregion) == 4:
                width = float(scrollregion[2])
                height = float(scrollregion[3])
                
                if width > 0 and height > 0:
                    # Convert pixel offsets to scroll fractions
                    dx_frac = dx / width
                    dy_frac = dy / height
                    
                    # Get current scroll position
                    x_scroll = self.canvas.canvasx(0) / width
                    y_scroll = self.canvas.canvasy(0) / height
                    
                    # Apply scroll adjustment
                    self.canvas.xview_moveto(x_scroll + dx_frac)
                    self.canvas.yview_moveto(y_scroll + dy_frac)
    
    # ------------- Image / Project Loading -------------
    def open_image(self):
        """Open a single image (non‑project mode)."""
        filetypes = (("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*"))
        filepath = filedialog.askopenfilename(initialdir=self.last_project_dir, title="Open Image", filetypes=filetypes)
        if not filepath:
            return
        self.last_project_dir = os.path.dirname(filepath)
        self.image_path = filepath
        try:
            self.original_image = Image.open(filepath)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}")
            return
        self.zoom = 1.0
        self.results = []
        self.results_tree.delete(*self.results_tree.get_children())
        self.update_canvas_image()
        
        # Auto-fit to window after a short delay
        self.root.after(100, self.fit_to_window)
        self.current_filename = os.path.basename(self.image_path)
        self.filename_info_label.config(text="Filename: " + self.current_filename)
    
    def load_current_image(self):
        if not self.project_data or not self.project_data["images"]:
            return
        current_entry = self.project_data["images"][self.project_data["current_index"]]
        images_folder = os.path.join(self.project_data["project_folder"], "images")
        image_path = os.path.join(images_folder, current_entry["filename"])
        self.image_path = image_path
        try:
            self.original_image = Image.open(image_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}")
            return
        self.zoom = 1.0
        
        # Load saved results.
        self.results = current_entry.get("results", [])
        self.results_tree.delete(*self.results_tree.get_children())
        for res in self.results:
            self.results_tree.insert("", "end", values=(res["id"], res["x"], res["Y"], res["angle"], res["rolling"]))
        self.update_canvas_image()
        
        # Auto-fit to window after a short delay
        self.root.after(100, self.fit_to_window)
        self.current_filename = os.path.basename(self.image_path)
        self.filename_info_label.config(text="Filename: " + self.current_filename)
        
        # Load simulation plot if it exists in labels folder
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        base = os.path.splitext(os.path.basename(self.image_path))[0]
        out_folder = self.project_data["project_folder"] if self.project_data else os.path.dirname(self.image_path)
        labels_folder = os.path.join(out_folder, "labels")
        simulation_file = os.path.join(labels_folder, base + "_simulation_plot.png")
        if os.path.exists(simulation_file):
            try:
                dist_img = Image.open(simulation_file)
                dist_img = dist_img.resize((400, 300), resample=resample_filter)
                self.distribution_image = ImageTk.PhotoImage(dist_img)
                lbl = ttk.Label(self.plot_frame, image=self.distribution_image)
                lbl.pack(fill=tk.BOTH, expand=True)
            except Exception as e:
                print("Error loading simulation plot:", e)
    
    def update_project_nav_label(self):
        if self.project_data:
            current = self.project_data["current_index"] + 1
            total = len(self.project_data["images"])
            self.project_nav_label.config(text=f"Image {current} of {total}")
        else:
            self.project_nav_label.config(text="No project loaded")
    
    # ------------- Project Management -------------
    def new_project(self):
        project_folder = filedialog.askdirectory(initialdir=self.last_project_dir, title="Select Folder for New Project")
        if not project_folder:
            return
        self.last_project_dir = project_folder
        images_folder = os.path.join(project_folder, "images")
        os.makedirs(images_folder, exist_ok=True)
        self.project_data = {
            "project_folder": project_folder,
            "images": [],
            "current_index": 0
        }
        self.save_project()
        self.canvas.delete("all")
        self.results = []
        self.results_tree.delete(*self.results_tree.get_children())
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        self.image_path = None
        self.current_filename = None
        self.filename_info_label.config(text="Filename: None")
        self.project_name = os.path.basename(project_folder)
        self.project_info_label.config(text="Project: " + self.project_name)
        self.update_project_nav_label()
        self.root.title(f"Leaf Angle Calculator - [Project: {self.project_name}]")
    
    def open_project(self):
        proj_file = filedialog.askopenfilename(initialdir=self.last_project_dir, title="Open Project File", filetypes=[("Project Files", "*.json")])
        if not proj_file:
            return
        self.last_project_dir = os.path.dirname(proj_file)
        with open(proj_file, "r") as f:
            self.project_data = json.load(f)
        if "current_index" not in self.project_data:
            self.project_data["current_index"] = 0
        self.project_name = os.path.basename(self.project_data["project_folder"])
        self.project_info_label.config(text="Project: " + self.project_name)
        self.root.title(f"Leaf Angle Calculator - [Project: {self.project_name}]")
        self.load_current_image()
        self.update_project_nav_label()
    
    def add_image(self):
        if not self.project_data:
            messagebox.showwarning("No Project", "Please create or open a project first.")
            return
        filepath = filedialog.askopenfilename(initialdir=self.last_project_dir, title="Select Image",
                                              filetypes=(("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")))
        if not filepath:
            return
        self.last_project_dir = os.path.dirname(filepath)
        images_folder = os.path.join(self.project_data["project_folder"], "images")
        basename = os.path.basename(filepath)
        dest_path = os.path.join(images_folder, basename)
        shutil.copy(filepath, dest_path)
        new_image_entry = {"filename": basename, "results": []}
        self.project_data["images"].append(new_image_entry)
        self.project_data["current_index"] = len(self.project_data["images"]) - 1
        self.save_project()
        self.load_current_image()
        self.update_project_nav_label()
    
    def prev_image(self):
        if not self.project_data or not self.project_data["images"]:
            return
        if self.project_data["current_index"] > 0:
            self.project_data["current_index"] -= 1
            self.save_project()
            self.load_current_image()
            self.update_project_nav_label()
    
    def next_image(self):
        if not self.project_data or not self.project_data["images"]:
            return
        if self.project_data["current_index"] < len(self.project_data["images"]) - 1:
            self.project_data["current_index"] += 1
            self.save_project()
            self.load_current_image()
            self.update_project_nav_label()
    
    def save_project(self):
        if not self.project_data:
            return
        proj_file = os.path.join(self.project_data["project_folder"], "project.json")
        with open(proj_file, "w") as f:
            json.dump(self.project_data, f, indent=4)
    
    # ------------- Point Creation & Leaf Calculation -------------
    def on_canvas_click(self, event):
        """Handle mouse click to create a point"""
        if not self.original_image:
            return
        
        # Convert click coordinates to original image coordinates
        x_original = event.x / self.zoom
        y_original = event.y / self.zoom
        
        # Set default values for angle and rolling
        angle = 0.0
        rolling = 0.0
        
        # Create new result entry
        new_id = len(self.results) + 1
        result = {
            "id": new_id,
            "x": round(x_original, 2),
            "Y": round(y_original, 2),
            "angle": angle,
            "rolling": rolling,
            "point": (x_original, y_original)
        }
        
        self.results.append(result)
        self.results_tree.insert("", "end", values=(result["id"], result["x"], result["Y"], result["angle"], result["rolling"]))
        
        # Save to project if in project mode
        if self.project_data:
            self.project_data["images"][self.project_data["current_index"]]["results"] = self.results
            self.save_project()
        
        # Draw the point on canvas
        self.draw_point(event.x, event.y, new_id)
    
    def draw_point(self, x, y, point_id):
        """Draw a point (circle) on the canvas"""
        radius = 4
        circle_id = self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, 
                                          fill="blue", outline="darkblue", width=2)
        # Add text label with ID
        text_id = self.canvas.create_text(x, y - 15, text=str(point_id), fill="blue", font=("Arial", 10, "bold"))
        self.point_ids.extend([circle_id, text_id])
    
    
    def on_treeview_double_click(self, event):
        region = self.results_tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        row_id = self.results_tree.identify_row(event.y)
        column = self.results_tree.identify_column(event.x)
        # Allow editing angle (#4) or rolling (#5) columns
        if column not in ["#4", "#5"]:
            return
        x, y, width, height = self.results_tree.bbox(row_id, column)
        self.edit_entry = tk.Entry(self.results_tree)
        self.edit_entry.place(x=x, y=y, width=width, height=height)
        current_value = self.results_tree.set(row_id, column)
        self.edit_entry.insert(0, current_value)
        self.edit_entry.focus_set()
        self.edit_entry.bind("<FocusOut>", lambda e: self.save_edit(row_id, column))
        self.edit_entry.bind("<Return>", lambda e: self.save_edit(row_id, column))
    
    def save_edit(self, row_id, column):
        new_val = self.edit_entry.get()
        try:
            new_value = float(new_val)
            # Validate angle values
            if column == "#4" and (new_value < 0 or new_value > 90):
                raise ValueError("Angle must be between 0 and 90")
        except ValueError:
            error_msg = "Angle must be numeric and between 0 and 90." if column == "#4" else "Rolling must be numeric."
            messagebox.showerror("Invalid value", error_msg)
            self.edit_entry.destroy()
            return
        
        self.results_tree.set(row_id, column, round(new_value, 2))
        leaf_id = self.results_tree.set(row_id, "#1")
        
        # Update the corresponding result
        for res in self.results:
            if str(res["id"]) == str(leaf_id):
                if column == "#4":  # angle
                    res["angle"] = round(new_value, 2)
                elif column == "#5":  # rolling
                    res["rolling"] = round(new_value, 2)
                break
        
        if self.project_data:
            self.project_data["images"][self.project_data["current_index"]]["results"] = self.results
            self.save_project()
        self.edit_entry.destroy()
    
    def on_treeview_select(self, event):
        # Clear previous highlighting
        if self.highlighted_point_ids:
            for pid in self.highlighted_point_ids:
                self.canvas.delete(pid)
            self.highlighted_point_ids = []
        
        selection = self.results_tree.selection()
        if not selection:
            return
        item = selection[0]
        values = self.results_tree.item(item, "values")
        if not values:
            return
        leaf_id = values[0]
        try:
            leaf_id = int(leaf_id)
        except ValueError:
            pass
        
        # Highlight selected point
        for res in self.results:
            if res["id"] == leaf_id and "point" in res:
                x, y = res["point"]
                # Draw highlighted circle around the point
                radius = 8
                highlight_id = self.canvas.create_oval(x * self.zoom - radius, y * self.zoom - radius, 
                                                     x * self.zoom + radius, y * self.zoom + radius, 
                                                     outline="yellow", width=3, fill="")
                self.highlighted_point_ids.append(highlight_id)
                break
    
    def delete_selected_leaf(self):
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "No leaf selected!")
            return
        item = selection[0]
        values = self.results_tree.item(item, "values")
        if not values:
            return
        leaf_id = int(values[0])
        
        # Remove the selected result
        self.results = [res for res in self.results if res["id"] != leaf_id]
        
        # Reassign sequential IDs
        for idx, res in enumerate(self.results, start=1):
            res["id"] = idx
        
        # Update treeview
        self.results_tree.delete(*self.results_tree.get_children())
        for res in self.results:
            self.results_tree.insert("", "end", values=(res["id"], res["x"], res["Y"], res["angle"], res["rolling"]))
        
        if self.project_data:
            self.project_data["images"][self.project_data["current_index"]]["results"] = self.results
            self.save_project()
        
        # Refresh the canvas to re-draw remaining points
        self.update_canvas_image()
    
    
    def save_csv(self):
        if self.project_data:
            base = os.path.splitext(os.path.basename(self.image_path))[0]
            # Create labels directory
            labels_dir = os.path.join(self.project_data["project_folder"], "labels")
            os.makedirs(labels_dir, exist_ok=True)
            csv_filename = os.path.join(labels_dir, base + ".csv")
            results_to_save = self.results
        else:
            if not self.image_path:
                messagebox.showwarning("Warning", "No image loaded!")
                return
            # Create labels directory next to image
            image_dir = os.path.dirname(self.image_path)
            labels_dir = os.path.join(image_dir, "labels")
            os.makedirs(labels_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(self.image_path))[0]
            csv_filename = os.path.join(labels_dir, base + ".csv")
            results_to_save = self.results
        
        if not results_to_save:
            messagebox.showwarning("Warning", "No data to save!")
            return
            
        try:
            # Save CSV file
            with open(csv_filename, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["id", "x", "Y", "angle", "rolling"])
                for res in results_to_save:
                    writer.writerow([res["id"], res["x"], res["Y"], res["angle"], res["rolling"]])
            
            # Extract angles for simulation
            angles = np.array([res["angle"] for res in results_to_save if res["angle"] > 0])
            
            success_message = f"Data saved to:\n{csv_filename}"
            
            # Generate simulations if we have enough angle data
            if len(angles) >= 3:
                try:
                    sim_path, plot_path = self.beta_fitter.process_angles(angles, csv_filename)
                    if sim_path and plot_path:
                        success_message += f"\n\nSimulation files created in labels folder:"
                        success_message += f"\n• Simulations: {os.path.basename(sim_path)}"
                        success_message += f"\n• Plot: {os.path.basename(plot_path)}"
                    else:
                        success_message += "\n\nNote: Could not generate simulations (insufficient data or error)"
                except Exception as e:
                    success_message += f"\n\nSimulation error: {str(e)}"
            else:
                success_message += f"\n\nNote: Need at least 3 angle measurements for simulations (have {len(angles)})"
            
            messagebox.showinfo("Success", success_message)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV file:\n{e}")
    
    # ------------- Distribution Plot -----------------
    def plot_distribution(self):
        if self.project_data:
            data = np.array([res["angle"] for res in self.results if res["angle"] > 0])
        else:
            if not self.results:
                messagebox.showwarning("Warning", "No leaf data to plot!")
                return
            data = np.array([res["angle"] for res in self.results if res["angle"] > 0])
        
        if data.size < 3:
            messagebox.showwarning("Warning", "Need at least 3 angle measurements to plot distribution!")
            return
            
        try:
            # Use the beta fitter for consistency
            fit_result = self.beta_fitter.fit_mle(data)
            
            # Create the plot
            fig = Figure(figsize=(4, 3), dpi=100)
            ax = fig.add_subplot(111)
            
            # Plot histogram
            ax.hist(data, bins=np.arange(0, 95, 5), density=True, alpha=0.6, color='lightblue', label='Data')
            
            # Plot fitted distribution
            x = np.linspace(0, 90, 200)
            x_adj = np.clip((x - self.beta_fitter.ncp)/90.0, 0.00001, 0.99999)
            a, b = fit_result['estimate']
            y = stats.beta.pdf(x_adj, a, b) / 90.0
            
            # Calculate mean angle
            y_for_mean = stats.beta.pdf(np.linspace(0, 1, self.beta_fitter.angle_resolution) - self.beta_fitter.ncp/90.0, a, b)
            y_for_mean = np.nan_to_num(y_for_mean, nan=0.0, posinf=0.0, neginf=0.0)
            sum_y = np.sum(y_for_mean)
            if sum_y > 0:
                y_for_mean = y_for_mean / sum_y
            mean_angle = self.beta_fitter.mean_angle(y_for_mean)
            
            ax.plot(x, y, 'r-', lw=2, label=f'Beta fit (α={a:.2f}, β={b:.2f})')
            ax.set_title(f"Distribution of Inclination Angles (mean={mean_angle:.1f}°)")
            ax.set_xlabel("Inclination Angle (°)")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Remove any existing distribution plot widgets
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
            if self.distribution_canvas:
                self.distribution_canvas.get_tk_widget().destroy()
                self.distribution_canvas = None
                
            self.distribution_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            self.distribution_canvas.draw()
            self.distribution_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot distribution:\n{e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = LeafAngleLabelingApp(root)
    root.mainloop()
