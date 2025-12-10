from PIL import Image, ImageTk
import numpy as np
import os
import json
import datetime


import tkinter as tk
import tkinter.messagebox as tkMessageBox
import tkinter.filedialog as tkFileDialog
import tkinter.font as tkFont
import tkinter.ttk as ttk
from ttkthemes import ThemedStyle

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_interactions import zoom_factory, panhandler
from itertools import cycle
from pathlib import Path
import configparser

from ramses2.utils.utils import crop_to_aspect_ratio, pad_to_aspect_ratio
from ramses2.inference import predict, stream_predict
import ramses2.GUI.gui_utils as gui_utils


# local_path = Path(os.path.realpath(os.path.dirname(__file__)))
RES = 20.5
MODEL_PATH = Path("../../checkpoints/2048x3072/best-val-loss.pt")
DEFAULT_IMG_SIZE = 4096  # if one of the side of the image is larger it is resized to speed-up the display
COLORS = {"Ra": "dimgray", "Rb": "orange", "Rc": "lightblue", "Ru": "yellow", "Rg": "green", "X": "purple"}

CONFIG_PATH = os.path.expanduser("~") / Path(".config/ramses/ramses.ini")

configini = configparser.ConfigParser()
configini.read(CONFIG_PATH)


def update_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as configfile:
        config.write(configfile)


try:
    INPUT_FOLDER = Path(configini["config"]["input_folder"])
except KeyError:
    configini["config"] = {"input_folder": os.path.expanduser("~")}
    INPUT_FOLDER = os.path.expanduser("~")
    update_config(configini)


class SetParams(tk.Toplevel):

    def __init__(self, parent, params, **kwargs):
        super().__init__(parent)
        self.config(width=400, height=400)
        self.title("Properties")

        self.params = params
        self.tk_params = {
            "score_threshold": tk.DoubleVar(),
            "mask_threshold": tk.DoubleVar(),
            "nms_threshold": tk.DoubleVar(),
            "crop_to_aspect_ratio": tk.BooleanVar(),
            "network_input_height": tk.IntVar(),
            "network_input_width": tk.IntVar(),
            # "from_stream": tk.BooleanVar(),
            "min_area": tk.IntVar(),
            "max_instances": tk.IntVar(),
            "max_view_size": tk.IntVar(),
        }

        for k, v in self.params.items():
            self.tk_params[k].set(v)

        ttk.Label(self, text="Network input size (height, width):").grid(column=0, row=0, sticky="EWSN", padx=5)
        ttk.Label(self, text="Crop to aspect_ratio:").grid(column=0, row=1, sticky="EWSN", padx=5)
        # ttk.Label(self, text="Continuous stream mode:").grid(column=0, row=2, sticky="EWSN", padx=5)
        ttk.Label(self, text="Score threshold:").grid(column=0, row=2, sticky="EWSN", padx=5)
        ttk.Label(self, text="mask threshold:").grid(column=0, row=3, sticky="EWSN", padx=5)
        ttk.Label(self, text="NMS threshold:").grid(column=0, row=4, sticky="EWSN", padx=5)
        ttk.Label(self, text="Delete objects with area <= ").grid(column=0, row=5, sticky="EWSN", padx=5)
        ttk.Label(self, text="Max number of detections:").grid(column=0, row=6, sticky="EWSN", padx=5)
        ttk.Label(self, text="Maxsize of saved segmentation images:").grid(column=0, row=7, sticky="EWSN", padx=5)

        tk.Entry(master=self, textvariable=self.tk_params["network_input_height"]).grid(
            column=1, row=0, sticky="EWSN", padx=5
        )
        tk.Entry(master=self, textvariable=self.tk_params["network_input_width"]).grid(
            column=2, row=0, sticky="EWSN", padx=5
        )
        tk.Radiobutton(self, text="False", variable=self.tk_params["crop_to_aspect_ratio"], value=False).grid(
            column=1, row=1, sticky="EWSN", padx=5
        )
        tk.Radiobutton(self, text="True", variable=self.tk_params["crop_to_aspect_ratio"], value=True).grid(
            column=2, row=1, sticky="EWSN", padx=5
        )
        # tk.Radiobutton(self, text="False", variable=self.tk_params["from_stream"], value=False).grid(
        #     column=1, row=2, sticky="EWSN", padx=5
        # )
        # tk.Radiobutton(self, text="True", variable=self.tk_params["from_stream"], value=True).grid(
        #     column=2, row=2, sticky="EWSN", padx=5
        # )
        tk.Entry(master=self, textvariable=self.tk_params["score_threshold"]).grid(
            column=1, row=2, sticky="EWSN", padx=5
        )
        tk.Entry(master=self, textvariable=self.tk_params["mask_threshold"]).grid(
            column=1, row=3, sticky="EWSN", padx=5
        )
        tk.Entry(master=self, textvariable=self.tk_params["nms_threshold"]).grid(column=1, row=4, sticky="EWSN", padx=5)
        tk.Entry(master=self, textvariable=self.tk_params["min_area"]).grid(column=1, row=5, sticky="EWSN", padx=5)
        tk.Entry(master=self, textvariable=self.tk_params["max_instances"]).grid(column=1, row=6, sticky="EWSN", padx=5)
        tk.Entry(master=self, textvariable=self.tk_params["max_view_size"]).grid(column=1, row=7, sticky="EWSN", padx=5)

        self.save_button = ttk.Button(self, text="Save", command=lambda: self.save())
        self.save_button.grid(column=0, row=8, sticky="EWSN", padx=5)

        self.button_close = ttk.Button(self, text="Close without saving", command=self.close)
        self.button_close.grid(column=1, row=8, columnspan=2, sticky="EWSN", padx=5)

        self.focus()
        self.grab_set()

    def save(self):
        for k, v in self.tk_params.items():
            self.params[k] = v.get()
        self.destroy()
        self.update()

    def close(self):
        self.destroy()
        self.update()


class ramsesGUI:
    def __init__(self):
        super(ramsesGUI, self).__init__()
        self.root = tk.Tk()

        self.root.wm_title("Automatic Recycled Aggregates Characterization")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        # style = ttk.Style(self.root)
        style = ThemedStyle(self.root)
        style.configure(".", font=("Helvetica", 12))
        style.theme_use("scidblue")

        # self.bold_font = tkFont.nametofont("TkTextFont").copy()
        # self.bold_font.configure(weight="bold")
        self.root.geometry("1000x600")
        # self.style = ttk.Style()
        self.treeview_style = ttk.Style(self.root)
        self.treeview_style.configure("My.Treeview", font=("Helvetica", 12))
        self.treeview_style.configure("My.Treeview.Heading", background="grey", font=("Helvetica", 12, "bold"))

        # Variables
        self.input_dir = INPUT_FOLDER
        self.input_dir_var = tk.StringVar()
        self.input_dir_var.set(str(self.input_dir))
        self.resolution = tk.DoubleVar()
        self.resolution.set(RES)
        self.seg_button_text = tk.StringVar()
        self.mode = tk.StringVar()
        self.mode.set("Gestion des objets \n touchants les bords :\n Non")
        self.border_detection = False
        self.parameters = {
            "score_threshold": 0.5,
            "mask_threshold": 0.5,
            "nms_threshold": 0.25,
            "crop_to_aspect_ratio": True,
            "network_input_height": 2048,
            "network_input_width": 3072,
            # "from_stream": False,
            "min_area": 16,
            "max_instances": 768,
            "max_view_size": DEFAULT_IMG_SIZE,
        }
        # Init other variables
        self.reset()

        # Menubar: File
        self.menubar = tk.Menu(self.root)
        filemenu = tk.Menu(self.menubar, tearoff=0)
        filemenu.add_command(label="Load model", command=lambda: self.gui_load_model())
        filemenu.add_command(label="Quit", command=self.close)
        self.menubar.add_cascade(label="File", menu=filemenu)

        propmenu = tk.Menu(self.menubar, tearoff=0)
        propmenu.add_command(label="parameters", command=lambda: self.run_params_window())
        self.menubar.add_cascade(label="Configure", menu=propmenu)

        self.root.config(menu=self.menubar)

        # Input folder selection and resolution
        Y = 0.1
        self.input_frame = ttk.Frame(self.root)
        self.input_frame.place(relx=0, rely=0, relwidth=0.8, relheight=Y)  # (side=tk.TOP, fill=tk.X, expand=1)

        self.input_dir_button = ttk.Button(
            self.input_frame, text="Choix du répertoire", command=lambda: self.select_input_dir()
        )
        self.input_dir_button.grid(column=0, row=0, sticky="EWSN", padx=5)
        self.resolution_label = ttk.Label(master=self.input_frame, text="Resolution (pixel/mm)")
        self.resolution_label.grid(column=1, row=0, sticky="EWSN", padx=5)

        self.input_dir_label = tk.Entry(
            self.input_frame, textvariable=self.input_dir_var, width=30, state="disabled", justify=tk.RIGHT
        )
        self.input_dir_label.configure(disabledbackground="white", disabledforeground="black", selectbackground="gray")
        self.input_dir_label.grid(column=0, row=1, sticky="EWSN", padx=5)
        self.resolution_entry = tk.Entry(master=self.input_frame, textvariable=self.resolution, justify=tk.RIGHT)
        self.resolution_entry.grid(column=1, row=1, sticky="EWSN", padx=5)

        self.pred_button = ttk.Button(self.input_frame, width=15, text="Predict", command=lambda: self.run_inference())
        self.pred_button.grid(column=2, row=0, rowspan=2, sticky="EWSN", padx=5)

        # Notebook and tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.place(relx=0, rely=Y, relheight=1 - Y, relwidth=1)  # pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.root.update()

        self.image_tab = ttk.Frame(self.notebook)  # View image, segment, predict class and mass
        self.classes_tab = ttk.Frame(self.notebook)  # View defined class
        self.granulometry_tab = ttk.Frame(self.notebook)  # View granulometric curve
        self.classification_tab = ttk.Frame(self.notebook)  # view class prediction (mass)

        self.image_tab.pack(fill=tk.BOTH, expand=1)
        self.classification_tab.pack(fill=tk.BOTH, expand=1)
        self.granulometry_tab.pack(fill=tk.BOTH, expand=1)
        self.classes_tab.pack(fill=tk.BOTH, expand=1)

        self.notebook.add(self.image_tab, text="Image")
        self.notebook.add(self.classification_tab, text="Results")
        self.notebook.add(self.granulometry_tab, text="Granulometry")
        self.notebook.add(self.classes_tab, text="Class definition")

        # Image TAB
        # Canvas frame
        self.fig = plt.Figure()
        self.canvas_frame = ttk.Frame(self.image_tab)
        self.canvas_frame.place(relx=0, rely=0, relwidth=1, relheight=0.9)  # pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.img_canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.img_canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.img_canvas, self.canvas_frame)
        self.toolbar.update()
        self.img_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Buttons
        self.img_button_frame = ttk.Frame(self.image_tab)
        self.img_button_frame.place(x=0, rely=0.9, relheight=0.1, relwidth=1)
        self.view_seg_button = ttk.Button(
            self.img_button_frame, width=20, textvariable=self.seg_button_text, command=lambda: self.switch_view()
        )
        self.view_seg_button.pack(side=tk.LEFT, fill=None, expand=0)

        self.view_next_button = ttk.Button(
            self.img_button_frame, width=20, text="view previous image", command=lambda: self.view_prev()
        )

        self.view_next_button.pack(side=tk.LEFT, fill=None, expand=0)
        self.view_next_button = ttk.Button(
            self.img_button_frame, width=20, text="view next image", command=lambda: self.view_next()
        )
        self.view_next_button.pack(side=tk.LEFT, fill=None, expand=0)

        # Classification TAB - fig and table
        self.classification_fig = plt.Figure()
        self.classification_frame = ttk.Frame(self.classification_tab)
        self.classification_frame.place(relx=0, rely=0, relwidth=1, relheight=0.8)
        self.classification_canvas = FigureCanvasTkAgg(self.classification_fig, master=self.classification_frame)
        self.classification_canvas.draw()
        self.class_toolbar = NavigationToolbar2Tk(self.classification_canvas, self.classification_frame)
        self.class_toolbar.update()
        self.classification_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.classification_table = ttk.Treeview(self.classification_tab, style="My.Treeview")
        self.classification_table.place(
            relx=0, rely=0.8, relwidth=1, relheight=0.2
        )  # .pack(side=tk.TOP, fill=tk.X, expand=1, pady=5)

        # Granulometry TAB - fig and table
        self.granulometry_fig = plt.Figure()
        self.granulometry_frame = ttk.Frame(self.granulometry_tab)
        self.granulometry_frame.place(
            relx=0, rely=0, relwidth=1, relheight=0.8
        )  # .pack(side=tk.TOP, fill=tk.BOTH, expand=1)  #
        self.granulometry_canvas = FigureCanvasTkAgg(self.granulometry_fig, master=self.granulometry_frame)
        self.granulometry_canvas.draw()
        self.granulometry_toolbar = NavigationToolbar2Tk(self.granulometry_canvas, self.granulometry_frame)
        self.granulometry_toolbar.update()
        self.granulometry_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.granulometry_table = ttk.Treeview(self.granulometry_tab, style="My.Treeview")
        self.granulometry_table.place(
            relx=0, rely=0.8, relwidth=1, relheight=0.2
        )  # .pack(side=tk.TOP, fill=tk.X, expand=1, pady=5)

        # class info TAB - list all classes

        self.class_description = tk.Text(
            master=self.classes_tab, bg="white", font=("Helvetica", 18), spacing1=7, spacing2=7
        )
        self.class_description.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        with open(os.path.join(os.path.realpath(os.path.dirname(__file__)), "classes.json"), "r") as jsonfile:
            self.classes = json.load(jsonfile)
        for classname, classdata in self.classes.items():
            # ttk.Label(master=self.classes_tab, bg='white', text=f"{classname}: {classdata['description']}").grid(row=i,column=0, sticky='W')
            self.class_description.insert(tk.END, f"{classname}: {classdata['description']}\n")
        self.class_description.config(state=tk.DISABLED)

        self.root.update()

        # Load models
        try:
            self.model, self.idx_to_cls = gui_utils.load_model(
                os.path.join(os.path.realpath(os.path.dirname(__file__)), MODEL_PATH)
            )
            print(self.idx_to_cls)
        except Exception as e:
            popup = tk.Toplevel(self.root)
            popup.title("Error")
            popup.attributes("-topmost", "true")
            ttk.Label(
                popup,
                text=f"Error opening model {os.path.join(os.path.realpath(os.path.dirname(__file__)), MODEL_PATH)}: {e}",
                font=("Helvetica", 16, "bold"),
            ).pack(side=tk.TOP, fil=tk.BOTH, expand=1, pady=10, padx=10)
            ttk.Button(popup, text="OK", command=popup.destroy).pack(pady=10)
            print(f"Error opening model {os.path.join(os.path.realpath(os.path.dirname(__file__)), MODEL_PATH)}: {e}")
            self.model, self.idx_to_cls = None, None

        # Show images if possible
        self.init_img_list()

    def run_params_window(self):
        SetParams(self.root, self.parameters)

    def gui_load_model(self):
        file = tkFileDialog.askopenfilename(parent=self.root, initialdir=os.path.realpath(os.path.dirname(__file__)))
        self.model, self.idx_to_cls = gui_utils.load_model(file)

    def view_next(self):
        self.img_pointer = self.next()

        if self.seg_mode:
            self.current_image_fn = self.img_list[self.img_pointer]
            self.img = self.open_image(os.path.join(self.input_dir, self.current_image_fn), plot=False)
            segname = os.path.join(self.detection_dir, "vizu", "VIZU-" + os.path.basename(self.current_image_fn))
            self.open_image(segname, plot=True)
        else:
            self.current_image_fn = self.img_list[self.img_pointer]
            self.img = self.open_image(os.path.join(self.input_dir, self.current_image_fn), plot=True)

    def view_prev(self):
        self.img_pointer = self.previous()

        if self.seg_mode:
            self.current_image_fn = self.img_list[self.img_pointer]
            self.img = self.open_image(os.path.join(self.input_dir, self.current_image_fn), plot=False)
            segname = os.path.join(self.detection_dir, "vizu", "VIZU-" + os.path.basename(self.current_image_fn))
            self.open_image(segname, plot=True)
        else:
            self.current_image_fn = self.img_list[self.img_pointer]
            self.img = self.open_image(os.path.join(self.input_dir, self.current_image_fn), plot=True)

    def switch_view(self):
        """plot the segmentation of the current image"""

        if self.detection_dir is not None and not self.seg_mode:
            segname = os.path.join(self.detection_dir, "vizu", "VIZU-" + os.path.basename(self.current_image_fn))
            self.open_image(segname, plot=True)

            # Update button properties
            self.seg_mode = not self.seg_mode
            if self.seg_mode:
                self.seg_button_text.set("View image")
            else:
                self.seg_button_text.set("View Segmentation")

        elif self.seg_mode:
            self.plotimg(self.img)
            # Update button properties
            self.seg_mode = not self.seg_mode
            if self.seg_mode:
                self.seg_button_text.set("View image")
            else:
                self.seg_button_text.set("View Segmentation")

    def plot_granulometry(self, annotations):
        """plot the granulometric curve of the current folder"""

        self.granulometry_fig.clear()
        self.gr_axes = self.granulometry_fig.add_subplot(111)
        self.granulometric_curve, self.gr_bins = gui_utils.compute_granulometry(
            annotations, "min_feret_diameter", self.resolution.get()
        )
        self.gr_axes.plot(self.gr_bins[1:], self.granulometric_curve, lw=2, marker=".")
        self.gr_axes.set_ylabel("Tamisats cumulés (%)")
        self.gr_axes.set_xlabel("Ouverture des tamis (mm)")
        self.gr_axes.set_xscale("log")
        self.gr_axes.set_ylim(0, 1)
        self.gr_axes.set_xlim(self.gr_bins[1], 100)
        self.gr_axes.xaxis.set_minor_locator(mtick.LogLocator(numticks=10000, subs="all"))
        # self.gr_axes.xaxis.set_minor_formatter(mtick.FormatStrFormatter("%.1f"))
        self.gr_axes.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))
        self.gr_axes.xaxis.set_minor_formatter(mtick.NullFormatter())
        self.gr_axes.grid(visible=True, which="major", axis="x", ls="-")
        self.gr_axes.grid(visible=True, which="minor", axis="x", ls="--")
        self.gr_axes.grid(visible=True, which="major", axis="y", ls="-")
        self.gr_axes.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        # self.granulometry_fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        zoom = zoom_factory(self.gr_axes)
        self.granulometry_canvas.draw()

        # Table
        self.granulometry_table.delete(*self.granulometry_table.get_children())
        colids = ["Tamis"]

        colids.extend([f"{i}" for i in list(self.gr_bins[:-1])])
        self.granulometry_table.configure(column=colids)
        self.granulometry_table["show"] = ""
        for i, c in enumerate(colids):
            if i == 0:
                self.granulometry_table.column(c, width=50, stretch=True, anchor="e")
            else:
                self.granulometry_table.column(c, width=25, stretch=True, anchor="e")

        self.granulometry_table.insert("", tk.END, values=colids, tag="even")
        val = ["Mass fraction (%)"]
        val.extend([f"{i:3.1f}" for i in list(self.granulometric_curve * 100)])
        self.granulometry_table.insert("", tk.END, values=val, tag="odd")
        self.granulometry_table.tag_configure("even", background="lightgrey", font=("Helvetica", 12, "bold"))
        self.root.update()

    def plot_classification(self, extended=False):
        """plot the granulometric curve of the current folder"""

        self.classification_fig.clear()
        self.classification_axes = self.classification_fig.add_subplot(111)
        fracs = self.EN933_preds["mass_fraction"]
        masses = self.EN933_preds["mass(g)"]
        labels = self.EN933_preds["class"]
        colors = [COLORS[c] for c in labels]
        self.classification_axes.pie(
            fracs,
            labels=labels,
            autopct="%2.1f%%",
            colors=colors,
            wedgeprops={"edgecolor": "black", "linewidth": 1, "antialiased": True},
        )
        # self.classification_fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        zoom = zoom_factory(self.classification_axes)
        self.classification_canvas.draw()

        # Table
        ext_labels = ["Class"]
        ext_labels.extend(labels)
        self.classification_table.delete(*self.classification_table.get_children())
        self.classification_table.configure(column=ext_labels)
        self.classification_table["show"] = ""
        for i, c in enumerate(ext_labels):
            # self.classification_table.heading(c, text=c)
            if i == 0:
                self.classification_table.column(c, width=100, stretch=False, anchor="e")
            else:
                self.classification_table.column(c, width=70, stretch=False, anchor="e")
        self.classification_table.insert("", tk.END, values=ext_labels, tag="headers")
        val = ["Fraction(%)"]
        val.extend([f"{i*100:4.1f}" for i in fracs])
        mval = ["Mass(g)"]
        mval.extend([f"{i:4.2f}" for i in masses])
        self.classification_table.insert("", tk.END, values=val, tag="even")
        self.classification_table.insert("", tk.END, values=mval, tag="odd")
        self.classification_table.tag_configure("headers", background="lightgrey", font=("Helvetica", 12, "bold"))
        self.classification_table.tag_configure("odd", background="lightgrey")

        self.root.update()

    def run_inference(self):
        """Run detection and class prediction. Update other plots"""

        if len(self.img_list) <= 0:
            popup = tk.Toplevel(self.root)
            popup.title("Warning")
            popup.attributes("-topmost", "true")
            ttk.Label(popup, text="No image found in selected folder", font=("Helvetica", 16, "bold")).pack(
                side=tk.TOP, fil=tk.BOTH, expand=1, pady=10, padx=10
            )
            ttk.Button(popup, text="OK", command=popup.destroy).pack(pady=(0, 10))
            return

        if self.model is None:
            popup = tk.Toplevel(self.root)
            popup.title(f"Warning")
            popup.attributes("-topmost", "true")
            ttk.Label(popup, text=f"Please load a model!", font=("Helvetica", 16, "bold")).pack(
                side=tk.TOP, fil=tk.BOTH, expand=1, pady=10, padx=10
            )
            ttk.Button(popup, text="OK", command=popup.destroy).pack(pady=(0, 10))
            return

        popup = tk.Toplevel(self.root)
        popup.title("")
        popup.attributes("-topmost", "true")
        ttk.Label(popup, text=f"Processing images, please wait", font=("Helvetica", 16, "bold")).pack(
            side=tk.TOP, fil=tk.BOTH, expand=1, pady=10, padx=10
        )
        progress_text = tk.StringVar()
        progress_text.set("Detecting aggregates...")
        progess_label = ttk.Label(popup, textvariable=progress_text, font=("Helvetica", 16, "bold"))
        progess_label.pack(side=tk.TOP, fil=tk.BOTH, expand=1, pady=10, padx=10)
        self.root.update()
        self.root.after(100)

        self.detection_dir = os.path.join(
            self.input_dir, "detection-" + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        )

        thresholds = [
            self.parameters["score_threshold"],
            self.parameters["mask_threshold"],
            self.parameters["nms_threshold"],
        ]
        # if self.parameters["from_stream"]:
        #     results = stream_predict(
        #         output_dir=self.detection_dir,
        #         input_size=(self.parameters["network_input_height"], self.parameters["network_input_width"]),
        #         resolution=self.resolution.get(),
        #         input_dir=self.input_dir,
        #         model=self.model,
        #         idx_to_cls=self.idx_to_cls,
        #         thresholds=thresholds,
        #         crop_to_ar=self.parameters["crop_to_aspect_ratio"],
        #         create_coco_anns=False,
        #         max_detections=self.parameters["max_instances"],
        #         minarea=self.parameters["min_area"],
        #         subdirs=False,
        #         boundary_mode="thick",
        #         save_imgs="class",
        #         device="cuda:0",
        #     )

        # else:

        results = predict(
            output_dir=self.detection_dir,
            input_size=(self.parameters["network_input_height"], self.parameters["network_input_width"]),
            resolution=self.resolution.get(),
            input_dir=self.input_dir,
            model=self.model,
            idx_to_cls=self.idx_to_cls,
            thresholds=thresholds,
            crop_to_ar=self.parameters["crop_to_aspect_ratio"],
            create_coco_anns=False,
            max_detections=self.parameters["max_instances"],
            minarea=self.parameters["min_area"],
            subdirs=False,
            boundary_mode="thick",
            save_imgs="class",
            device="cuda:0",
        )
        self.annotations, self.extended_preds, self.EN933_preds = gui_utils.process_predictions(results)

        # Update class and granulometry plots
        progress_text.set("Computing granulometry...")
        progess_label.update()
        self.root.after(100)
        self.plot_granulometry(self.annotations)
        progress_text.set("Plotting results...")
        progess_label.update()
        self.root.after(100)
        self.plot_classification()
        popup.destroy()
        # self.plot_segmentation()

    def plotimg(self, img):
        self.fig.clear()
        # with plt.ioff:
        self.img_axes = self.fig.add_subplot(111)
        plot = self.img_axes.imshow(img, interpolation="none", origin="upper")
        plot.set_extent((0, img.shape[1], 0, img.shape[0]))
        self.img_axes.axis("off")
        self.fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        zoom = zoom_factory(self.img_axes)
        pan = panhandler(self.fig)
        self.img_canvas.draw()
        self.root.update()

    def open_image(self, filename, plot=True, lmax=DEFAULT_IMG_SIZE):
        # Open image filename and plot it if plot is True
        # Return image as a numpy array
        if lmax is None:
            lmax = DEFAULT_IMG_SIZE

        try:
            img = Image.open(filename)
        except Exception as e:
            popup = tk.Toplevel(self.root)
            popup.title("Error")
            popup.attributes("-topmost", "true")
            ttk.Label(popup, text=f"Error opening {filename}: {e}", font=("Helvetica", 16, "bold")).pack(
                side=tk.TOP, fil=tk.BOTH, expand=1, pady=10, padx=10
            )
            ttk.Button(popup, text="OK", command=popup.destroy).pack(pady=(0, 10))
            return

        l = np.max(img.size)
        if l > lmax:
            ratio = l / lmax
            nx, ny = img.size
            img = img.resize((int(nx // ratio), int(ny // ratio)))
        img = np.array(img)
        if plot:
            self.plotimg(img)
        return img

    def init_img_list(self):
        self.img_list = gui_utils.get_img_list(self.input_dir)
        self.img_pointer = 0
        # Plot the first image in list
        if len(self.img_list) > 0:
            self.current_image_fn = os.path.join(self.input_dir, self.img_list[0])
            self.img = self.open_image(self.current_image_fn)
            self.reset()

    def select_input_dir(self):
        self.input_dir = tkFileDialog.askdirectory(parent=self.root, initialdir=INPUT_FOLDER)

        if self.input_dir:
            self.input_dir_var.set(self.input_dir)
            configini["config"]["input_folder"] = self.input_dir
            update_config(configini)
            self.init_img_list()

    def next(self):
        return self.img_pointer + 1 if self.img_pointer + 1 < len(self.img_list) else 0

    def previous(self):
        return self.img_pointer - 1 if self.img_pointer > 0 else len(self.img_list) - 1

    def reset(self):
        # Reset variables
        self.seg_button_text.set("View Segmentation")
        self.seg_mode = False
        self.annotations, self.extended_preds, self.EN933_preds = (None, None, None)
        self.segmentation_image = None
        self.detection_dir = None

    def save(self):
        pass

    def close(self):
        self.root.destroy()


if __name__ == "__main__":
    GUI = ramsesGUI()
    GUI.root.mainloop()
