"""Interactive Tkinter GUI for RAMSES2 inference and result visualization.

The window lets the user select an input folder, configure the inference
parameters, run instance-segmentation + mass prediction over every image in the
folder, and then inspect the results across four notebook tabs:

    * ``Image``            -- browse the raw images or their segmentation overlays.
    * ``Results``          -- EN 933-11 mass pie chart and per-class summary table.
    * ``Granulometry``     -- cumulative (mass) granulometric curve and value table.
    * ``Class definition`` -- human-readable description of the recycled-aggregate classes.

The application state (thresholds, network input size, device, ...) is persisted
in an ``.ini`` file managed by :class:`ConfigHandler`.
"""

from __future__ import annotations

import configparser
import datetime
import json
import os
import platform
from pathlib import Path

import numpy as np
import tkinter as tk
import tkinter.filedialog as tkFileDialog
import tkinter.messagebox as tkMessageBox
import tkinter.ttk as ttk
from PIL import Image
from ttkthemes import ThemedStyle

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_interactions import panhandler, zoom_factory

from ramses2.inference import predict
import ramses2.GUI.gui_utils as gui_utils

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Default pixel resolution of the input images, in pixels per millimetre.
RES: float = 20.5

#: Path of the model checkpoint used at startup (relative to this file).
MODEL_PATH: Path = Path("../../checkpoints/2048x3072/best-val-loss.pt")

#: If the longest side of an image exceeds this value (px), it is downscaled for
#: display so that panning and zooming stay fast.
DEFAULT_IMG_SIZE: int = 4096

#: EN 933-11 super-class -> colour used for the results pie chart.
COLORS: dict[str, str] = {
    "Ra": "dimgray",
    "Rb": "orange",
    "Rc": "lightblue",
    "Ru": "yellow",
    "Rg": "green",
    "X": "purple",
}

#: Location of the persisted configuration file.
CONFIG_PATH: Path = Path.home() / ".config" / "ramses" / "ramses.ini"

def _select_app_font() -> str:
    """Return the UI font family that renders with anti-aliasing on the current OS.

    Each platform is mapped to its native, well-hinted UI typeface so that Tk
    produces clean, subpixel-antialiased text:

    * macOS   -> ``Helvetica Neue`` (modern system UI font, CoreText anti-aliasing)
    * Windows -> ``Segoe UI``       (native UI font, ClearType subpixel anti-aliasing)
    * Linux   -> ``DejaVu Sans``    (default well-hinted TrueType, X anti-aliasing)

    The previous hard-coded "Helvetica" name was not a real, well-hinted font on
    Linux and was silently substituted, causing the jagged text the user reported.

    Returns:
        The font-family name to use throughout the GUI.
    """
    system = platform.system()
    if system == "Darwin":
        return "Helvetica Neue"
    if system == "Windows":
        return "Segoe UI"
    return "DejaVu Sans"


#: Application-wide font family, chosen per OS (see :func:`_select_app_font`).
APP_FONT: str = _select_app_font()


class ConfigHandler:
    """Load, validate and persist the GUI configuration file (``.ini``).

    The file is split into two sections:

    * ``[CONFIG]``   -- application settings (input folder).
    * ``[SETTINGS]`` -- inference parameters (thresholds, input size, device, ...).

    A missing file or a missing section is rebuilt from defaults on first access.
    """

    #: Default inference parameters used when the file is missing or corrupt.
    DEFAULT_PARAMETERS: dict[str, object] = {
        "score_threshold": 0.5,
        "mask_threshold": 0.5,
        "nms_threshold": 0.25,
        "crop_to_aspect_ratio": True,
        "network_input_height": 2048,
        "network_input_width": 3072,
        "min_area": 16,
        "max_instances": 768,
        "max_view_size": 1024,
        "device": "cuda",
    }

    def __init__(self, filename: Path | str | None = CONFIG_PATH) -> None:
        """Initialise the handler and load the configuration from ``filename``.

        Args:
            filename: Path to the ``.ini`` file. ``None`` disables persistence.
        """
        self.filename = filename
        self.config: dict[str, str] = {"input_folder": os.path.expanduser("~")}
        self.parameters: dict[str, object] = dict(self.DEFAULT_PARAMETERS)
        if filename is not None:
            self.load_parameters()

    def save_parameters(self) -> None:
        """Write the current configuration to disk.

        The parent directory is created if it does not exist yet.
        """
        config = configparser.ConfigParser()
        config["SETTINGS"] = {k: str(v) for k, v in self.parameters.items()}
        config["CONFIG"] = {k: str(v) for k, v in self.config.items()}

        parent = os.path.dirname(self.filename)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(self.filename, "w") as configfile:
            config.write(configfile)
        print(f"Configuration created/updated: {self.filename}")

    def load_parameters(self) -> None:
        """Load the configuration from disk, rebuilding it if it is incomplete.

        Every value is parsed with an explicit type. A single corrupt entry keeps
        its default instead of leaving the parameters only partially updated.
        """
        config = configparser.ConfigParser()

        if not os.path.exists(self.filename):
            print("Config file missing. Generating defaults...")
            self.save_parameters()
            return

        config.read(self.filename)

        if "SETTINGS" not in config or "CONFIG" not in config:
            print("Config file incomplete. Rebuilding...")
            self.save_parameters()
            return

        # [CONFIG] section
        self.config["input_folder"] = config["CONFIG"].get("input_folder", os.path.expanduser("~"))

        # [SETTINGS] section -- parsed into a temporary dict, then applied only if
        # every value is valid (all-or-nothing update).
        s = config["SETTINGS"]
        parsed: dict[str, object] = {}
        try:
            parsed["score_threshold"] = s.getfloat("score_threshold", 0.5)
            parsed["mask_threshold"] = s.getfloat("mask_threshold", 0.5)
            parsed["nms_threshold"] = s.getfloat("nms_threshold", 0.25)
            parsed["crop_to_aspect_ratio"] = s.getboolean("crop_to_aspect_ratio", True)
            parsed["network_input_height"] = s.getint("network_input_height", 2048)
            parsed["network_input_width"] = s.getint("network_input_width", 3072)
            parsed["min_area"] = s.getint("min_area", 16)
            parsed["max_instances"] = s.getint("max_instances", 768)
            parsed["max_view_size"] = s.getint("max_view_size", 1024)
            parsed["device"] = s.get("device", "cuda")
        except ValueError as exc:
            print(f"Error parsing [SETTINGS]: {exc}. Using defaults for this section.")
            return

        self.parameters.update(parsed)
        print("Configuration loaded successfully.")


class SetParams(tk.Toplevel):
    """Modal dialog to edit the inference parameters.

    Values are read from -- and written back to -- the shared ``parameters``
    dictionary of the parent :class:`ConfigHandler`, then persisted to disk.
    """

    def __init__(self, parent: tk.Misc, params: dict, config_manager: ConfigHandler, **kwargs) -> None:
        """Build the dialog.

        Args:
            parent: The parent Tk widget (used for positioning and modality).
            params: The shared parameters dictionary to edit in place.
            config_manager: The :class:`ConfigHandler` used to persist changes.
        """
        super().__init__(parent, **kwargs)
        self.config(width=400, height=420)
        self.title("Properties")
        self.config_manager = config_manager
        self.params = params

        self.tk_params: dict[str, tk.Variable] = {
            "score_threshold": tk.DoubleVar(),
            "mask_threshold": tk.DoubleVar(),
            "nms_threshold": tk.DoubleVar(),
            "crop_to_aspect_ratio": tk.BooleanVar(),
            "network_input_height": tk.IntVar(),
            "network_input_width": tk.IntVar(),
            "min_area": tk.IntVar(),
            "max_instances": tk.IntVar(),
            "max_view_size": tk.IntVar(),
            "device": tk.StringVar(),
        }
        for key, value in self.params.items():
            self.tk_params[key].set(value)

        self._build_layout()
        self.focus()
        self.grab_set()

    def _build_layout(self) -> None:
        """Create the labels, entries and buttons of the dialog."""
        labels = [
            "Network input size (height, width):",
            "Crop to aspect ratio:",
            "Score threshold:",
            "Mask threshold:",
            "NMS threshold:",
            "Remove objects with area <= (px):",
            "Max number of detections:",
            "Max size of saved segmentation images (px):",
        ]
        for row, text in enumerate(labels):
            ttk.Label(self, text=text).grid(column=0, row=row, sticky="E", padx=5, pady=2)

        # Network input height / width (row 0)
        tk.Entry(self, textvariable=self.tk_params["network_input_height"]).grid(column=1, row=0, sticky="EW", padx=5)
        tk.Entry(self, textvariable=self.tk_params["network_input_width"]).grid(column=2, row=0, sticky="EW", padx=5)

        # Crop to aspect ratio (row 1)
        tk.Radiobutton(self, text="False", variable=self.tk_params["crop_to_aspect_ratio"], value=False).grid(column=1, row=1, sticky="W", padx=5)
        tk.Radiobutton(self, text="True", variable=self.tk_params["crop_to_aspect_ratio"], value=True).grid(column=2, row=1, sticky="W", padx=5)

        # Scalar parameters (rows 2-7)
        scalar_rows = [
            (2, "score_threshold"),
            (3, "mask_threshold"),
            (4, "nms_threshold"),
            (5, "min_area"),
            (6, "max_instances"),
            (7, "max_view_size"),
        ]
        for row, key in scalar_rows:
            tk.Entry(self, textvariable=self.tk_params[key]).grid(column=1, row=row, sticky="EW", padx=5)

        # Device (row 8)
        tk.Radiobutton(self, text="CPU", variable=self.tk_params["device"], value="cpu").grid(column=1, row=8, sticky="W", padx=5)
        tk.Radiobutton(self, text="GPU", variable=self.tk_params["device"], value="cuda").grid(column=2, row=8, sticky="W", padx=5)

        # Save / close (row 9)
        ttk.Button(self, text="Save", command=self.save).grid(column=0, row=9, sticky="W", padx=5, pady=5)
        ttk.Button(self, text="Close without saving", command=self.close).grid(column=1, row=9, columnspan=2, sticky="EW", padx=5, pady=5)

    def save(self) -> None:
        """Read the widget values, persist them and close the dialog.

        If any numeric field holds an invalid value, a warning is shown and the
        dialog stays open so the user can correct the entry.
        """
        try:
            values = {key: var.get() for key, var in self.tk_params.items()}
        except tk.TclError as exc:
            tkMessageBox.showwarning("Invalid input", f"Please check the numeric fields.\n{exc}", parent=self)
            return
        self.params.update(values)
        self.config_manager.save_parameters()
        self.destroy()

    def close(self) -> None:
        """Close the dialog without saving any change."""
        self.destroy()


class ramsesGUI:
    """Main application window for RAMSES2.

    Builds the Tkinter interface, wires the inference pipeline to the GUI state,
    and renders the results (image/segmentation browser, EN 933-11 results,
    granulometric curve and class legend).
    """

    def __init__(self) -> None:
        """Create the root window and build the whole interface."""
        self.root: tk.Tk = tk.Tk()
        self.root.wm_title("Automatic Recycled Aggregates Characterization")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.geometry("1000x600")

        self._setup_style()
        self.config_manager = ConfigHandler(CONFIG_PATH)

        # Configuration / state variables
        self.input_dir: str = self.config_manager.config["input_folder"]
        self.input_dir_var = tk.StringVar(value=str(self.input_dir))
        self.resolution = tk.DoubleVar(value=RES)
        self.seg_button_text = tk.StringVar(value="View Segmentation")
        self.nb_aggregates_var = tk.StringVar(value="n/a")
        self.parameters: dict = self.config_manager.parameters

        # Image browser state
        self.img_list: list[str] = []
        self.img_pointer: int = 0
        self.current_image_fn: str | None = None
        self.img: np.ndarray | None = None
        self.seg_mode: bool = False
        self.detection_dir: str | None = None

        # Inference results
        self.annotations = None
        self.extended_preds = None
        self.EN933_preds = None

        # Build the interface
        self._build_menubar()
        self._build_input_frame()
        self._build_notebook()
        self.root.update()

        # Load the default model, then display the first image
        self._load_model_default()
        self.init_img_list()

    # ------------------------------------------------------------------
    # Interface construction helpers
    # ------------------------------------------------------------------
    def _setup_style(self) -> None:
        """Configure the ttk theme and the fonts used across the interface."""
        style = ThemedStyle(self.root)
        style.configure(".", font=(APP_FONT, 12))
        style.theme_use("scidblue")
        tree_style = ttk.Style(self.root)
        tree_style.configure("My.Treeview", font=(APP_FONT, 12))
        tree_style.configure("My.Treeview.Heading", background="grey", font=(APP_FONT, 12, "bold"))

    def _build_menubar(self) -> None:
        """Create the application menu bar (File / Configure)."""
        self.menubar = tk.Menu(self.root)

        filemenu = tk.Menu(self.menubar, tearoff=0)
        filemenu.add_command(label="Load model", command=self.gui_load_model)
        filemenu.add_command(label="Quit", command=self.close)
        self.menubar.add_cascade(label="File", menu=filemenu)

        configmenu = tk.Menu(self.menubar, tearoff=0)
        configmenu.add_command(label="Parameters", command=self.run_params_window)
        self.menubar.add_cascade(label="Configure", menu=configmenu)

        self.root.config(menu=self.menubar)

    def _build_input_frame(self) -> None:
        """Create the top frame with folder selection, resolution and controls."""
        self.input_frame = ttk.Frame(self.root)
        self.input_frame.place(relx=0, rely=0, relwidth=1, relheight=0.1)

        ttk.Button(self.input_frame, text="Select folder", command=self.select_input_dir).grid(column=0, row=0, sticky="EW", padx=5)
        ttk.Label(self.input_frame, text="Resolution (pixels/mm)").grid(column=1, row=0, sticky="EW", padx=5)

        dir_entry = tk.Entry(self.input_frame, textvariable=self.input_dir_var, width=30, state="disabled", justify=tk.RIGHT)
        dir_entry.configure(disabledbackground="white", disabledforeground="black", selectbackground="gray")
        dir_entry.grid(column=0, row=1, sticky="EW", padx=5)

        tk.Entry(self.input_frame, textvariable=self.resolution, justify=tk.RIGHT).grid(column=1, row=1, sticky="EW", padx=5)

        ttk.Button(self.input_frame, width=15, text="Predict", command=self.run_inference).grid(column=2, row=0, rowspan=2, sticky="EW", padx=5, ipadx=10)

        ttk.Label(self.input_frame, text="Detected Aggregates", anchor="center", relief="solid", padding=(5, 0)).grid(column=4, row=0, rowspan=2, sticky="EW", padx=(20, 5))
        ttk.Label(self.input_frame, textvariable=self.nb_aggregates_var, relief="solid", borderwidth=1, background="white", width=10, anchor="center").grid(column=5, row=0, rowspan=2, sticky="EW", padx=5)

    def _build_notebook(self) -> None:
        """Create the notebook and its four tabs."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.place(relx=0, rely=0.1, relwidth=1, relheight=0.9)

        self.image_tab = ttk.Frame(self.notebook)
        self.classification_tab = ttk.Frame(self.notebook)
        self.granulometry_tab = ttk.Frame(self.notebook)
        self.classes_tab = ttk.Frame(self.notebook)

        self._build_image_tab()
        self._build_classification_tab()
        self._build_granulometry_tab()
        self._build_classes_tab()

        self.notebook.add(self.image_tab, text="Image")
        self.notebook.add(self.classification_tab, text="Results")
        self.notebook.add(self.granulometry_tab, text="Granulometry")
        self.notebook.add(self.classes_tab, text="Class definition")

    def _build_image_tab(self) -> None:
        """Build the image / segmentation browser tab."""
        self.fig = plt.Figure()
        canvas_frame = ttk.Frame(self.image_tab)
        canvas_frame.place(relx=0, rely=0, relwidth=1, relheight=0.9)
        self.img_canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.img_canvas.draw()
        toolbar = NavigationToolbar2Tk(self.img_canvas, canvas_frame)
        toolbar.update()
        self.img_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        button_frame = ttk.Frame(self.image_tab)
        button_frame.place(x=0, rely=0.9, relwidth=1, relheight=0.1)
        ttk.Button(button_frame, width=20, textvariable=self.seg_button_text, command=self.switch_view).pack(side=tk.LEFT)
        ttk.Button(button_frame, width=20, text="Previous image", command=self.view_prev).pack(side=tk.LEFT)
        ttk.Button(button_frame, width=20, text="Next image", command=self.view_next).pack(side=tk.LEFT)

    def _build_classification_tab(self) -> None:
        """Build the results tab (mass-distribution chart + summary table).

        A selector at the top switches the displayed classes between the EN 933-11
        super-classes and every fine-grained class detected by the model. Both the
        pie chart and the summary table follow the selected mode.
        """
        # View-mode selector (top strip).
        selector = ttk.Frame(self.classification_tab)
        selector.place(relx=0, rely=0, relwidth=1, relheight=0.08)
        self.class_view_mode = tk.StringVar(value="en933")
        ttk.Label(selector, text="Classes:").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Radiobutton(selector, text="EN 933-11", variable=self.class_view_mode, value="en933", command=self.plot_classification).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(selector, text="All detected classes", variable=self.class_view_mode, value="all", command=self.plot_classification).pack(side=tk.LEFT, padx=5)

        # Mass-distribution chart.
        self.classification_fig = plt.Figure()
        frame = ttk.Frame(self.classification_tab)
        frame.place(relx=0, rely=0.08, relwidth=1, relheight=0.72)
        self.classification_canvas = FigureCanvasTkAgg(self.classification_fig, master=frame)
        self.classification_canvas.draw()
        toolbar = NavigationToolbar2Tk(self.classification_canvas, frame)
        toolbar.update()
        self.classification_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Summary table.
        self.classification_table = ttk.Treeview(self.classification_tab, style="My.Treeview")
        self.classification_table.place(relx=0, rely=0.8, relwidth=1, relheight=0.2)

    def _build_granulometry_tab(self) -> None:
        """Build the granulometry tab (curve + value table)."""
        self.granulometry_fig = plt.Figure()
        frame = ttk.Frame(self.granulometry_tab)
        frame.place(relx=0, rely=0, relwidth=1, relheight=0.8)
        self.granulometry_canvas = FigureCanvasTkAgg(self.granulometry_fig, master=frame)
        self.granulometry_canvas.draw()
        toolbar = NavigationToolbar2Tk(self.granulometry_canvas, frame)
        toolbar.update()
        self.granulometry_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.granulometry_table = ttk.Treeview(self.granulometry_tab, style="My.Treeview")
        self.granulometry_table.place(relx=0, rely=0.8, relwidth=1, relheight=0.2)

    def _build_classes_tab(self) -> None:
        """Build the class-legend tab from the local ``classes.json`` file."""
        self.class_description = tk.Text(
            master=self.classes_tab,
            bg="white",
            font=(APP_FONT, 14),
            spacing1=5,
            spacing2=5,
            padx=10,
            pady=10,
            wrap=tk.WORD,
        )
        self.class_description.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Tag styling: bold class names, normal description text.
        self.class_description.tag_configure("class_name", font=(APP_FONT, 14, "bold"))
        self.class_description.tag_configure("class_desc", font=(APP_FONT, 14))

        classes_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), "classes.json")
        with open(classes_path, "r", encoding="utf-8") as jsonfile:
            self.classes = json.load(jsonfile)
        # Display classes alphabetically by name rather than in their ID order.
        for classname, classdata in sorted(self.classes.items()):
            self.class_description.insert(tk.END, f"{classname}: ", ("class_name",))
            self.class_description.insert(tk.END, classdata["description"], ("class_desc",))
            self.class_description.insert(tk.END, "\n")
        self.class_description.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------
    def _load_model_default(self) -> None:
        """Load the default checkpoint; show an error dialog if it fails."""
        checkpoint = os.path.join(os.path.realpath(os.path.dirname(__file__)), MODEL_PATH)
        try:
            self.model, self.idx_to_cls = gui_utils.load_model(checkpoint)
            print(self.idx_to_cls)
        except Exception as exc:  # noqa: BLE001 -- any failure is shown to the user
            tkMessageBox.showerror("Error", f"Error opening model\n{checkpoint}\n\n{exc}", parent=self.root)
            print(f"Error opening model {checkpoint}: {exc}")
            self.model, self.idx_to_cls = None, None

    def gui_load_model(self) -> None:
        """Prompt for a checkpoint file and load it."""
        file = tkFileDialog.askopenfilename(parent=self.root, initialdir=os.path.realpath(os.path.dirname(__file__)))
        if not file:
            return
        self.model, self.idx_to_cls = gui_utils.load_model(file)

    # ------------------------------------------------------------------
    # Parameter dialog
    # ------------------------------------------------------------------
    def run_params_window(self) -> None:
        """Open the parameter-editing dialog."""
        SetParams(self.root, self.parameters, self.config_manager)

    # ------------------------------------------------------------------
    # Image navigation
    # ------------------------------------------------------------------
    @staticmethod
    def _next_pointer(img_pointer: int, length: int) -> int:
        """Return the pointer to the next image, wrapping around the list."""
        return (img_pointer + 1) % length

    @staticmethod
    def _prev_pointer(img_pointer: int, length: int) -> int:
        """Return the pointer to the previous image, wrapping around the list."""
        return (img_pointer - 1) % length

    def view_next(self) -> None:
        """Show the next image in the folder."""
        if not self.img_list:
            return
        self.img_pointer = self._next_pointer(self.img_pointer, len(self.img_list))
        self._display_current_image()

    def view_prev(self) -> None:
        """Show the previous image in the folder."""
        if not self.img_list:
            return
        self.img_pointer = self._prev_pointer(self.img_pointer, len(self.img_list))
        self._display_current_image()

    def _display_current_image(self) -> None:
        """Render the image at ``img_pointer`` (raw or segmentation overlay)."""
        self.current_image_fn = self.img_list[self.img_pointer]
        if self.seg_mode:
            self.img = self.open_image(self._full_path(self.current_image_fn), plot=False)
            self.open_image(self._segmentation_path(self.current_image_fn), plot=True)
        else:
            self.img = self.open_image(self._full_path(self.current_image_fn), plot=True)
        self._update_aggregate_count()

    def _update_aggregate_count(self) -> None:
        """Update the 'Detected Aggregates' counter for the current image."""
        if self.annotations is not None and self.current_image_fn is not None:
            mask = self.annotations["baseimg"] == os.path.basename(self.current_image_fn)
            self.nb_aggregates_var.set(str(int(mask.sum())))
        else:
            self.nb_aggregates_var.set("n/a")

    def _full_path(self, filename: str) -> str:
        """Return the path of an image filename inside the input folder."""
        return os.path.join(self.input_dir, filename)

    def _segmentation_path(self, filename: str) -> str:
        """Return the path of the saved segmentation overlay for an image.

        The overlay is always written by the inference pipeline as
        ``VIZU-<stem>.jpg`` inside ``<detection_dir>/vizu``, regardless of the
        original image extension.
        """
        stem = os.path.splitext(os.path.basename(filename))[0]
        return os.path.join(self.detection_dir, "vizu", f"VIZU-{stem}.jpg")

    def switch_view(self) -> None:
        """Toggle between the raw image and its segmentation overlay."""
        if self.seg_mode:
            # Currently showing the overlay -> go back to the raw image.
            self.plotimg(self.img)
            self.seg_mode = False
        elif self.detection_dir is not None and self.current_image_fn is not None:
            self.open_image(self._segmentation_path(self.current_image_fn), plot=True)
            self.seg_mode = True
        else:
            # No detection available yet; there is nothing to switch to.
            return
        self.seg_button_text.set("View image" if self.seg_mode else "View Segmentation")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def run_inference(self) -> None:
        """Run detection + class/mass prediction over the whole input folder.

        Shows a progress window, writes the results to a timestamped detection
        directory, then refreshes the results and granulometry tabs.
        """
        if not self.img_list:
            tkMessageBox.showwarning("Warning", "No image found in selected folder.", parent=self.root)
            return
        if self.model is None:
            tkMessageBox.showwarning("Warning", "Please load a model!", parent=self.root)
            return

        popup = tk.Toplevel(self.root)
        popup.title("Processing")
        popup.attributes("-topmost", "true")
        progress_text = tk.StringVar(value="Detecting aggregates...")
        ttk.Label(popup, textvariable=progress_text, font=(APP_FONT, 16, "bold")).pack(pady=15, padx=15)
        self.root.update()

        self.detection_dir = os.path.join(self.input_dir, "detection-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        thresholds = (
            self.parameters["score_threshold"],
            self.parameters["mask_threshold"],
            self.parameters["nms_threshold"],
        )

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
            device=self.parameters["device"],
        )

        self.annotations, self.extended_preds, self.EN933_preds = gui_utils.process_predictions(results)
        self._update_aggregate_count()

        progress_text.set("Computing granulometry...")
        self.root.update()
        self.plot_granulometry(self.annotations)

        progress_text.set("Plotting results...")
        self.root.update()
        self.plot_classification()

        popup.destroy()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_granulometry(self, annotations) -> None:
        """Draw the cumulative mass granulometric curve and its value table."""
        self.granulometry_fig.clear()
        axes = self.granulometry_fig.add_subplot(111)

        self.granulometric_curve, self.gr_bins = gui_utils.compute_granulometry(
            annotations, "min_feret_diameter", self.resolution.get()
        )
        curve, bins = self.granulometric_curve, self.gr_bins

        axes.plot(bins[1:], curve, lw=2, marker=".")
        axes.set_ylabel("Cumulative mass fraction (%)")
        axes.set_xlabel("Sieve opening (mm)")
        axes.set_xscale("log")
        axes.set_ylim(0, 1)
        axes.set_xlim(bins[1], max(float(bins[-1]), 100))
        axes.xaxis.set_minor_locator(mtick.LogLocator(numticks=10000, subs="all"))
        axes.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))
        axes.xaxis.set_minor_formatter(mtick.NullFormatter())
        axes.grid(visible=True, which="major", axis="x", ls="-")
        axes.grid(visible=True, which="minor", axis="x", ls="--")
        axes.grid(visible=True, which="major", axis="y", ls="-")
        axes.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        zoom_factory(axes)
        self.granulometry_canvas.draw()

        # Value table (one header row + one data row)
        self.granulometry_table.delete(*self.granulometry_table.get_children())
        header = ["Sieve"] + [f"{b}" for b in bins[:-1]]
        values = ["Mass fraction (%)"] + [f"{c * 100:3.1f}" for c in curve]
        self.granulometry_table.configure(columns=header)
        self.granulometry_table["show"] = ""
        for i, col in enumerate(header):
            self.granulometry_table.column(col, width=(50 if i == 0 else 25), stretch=True, anchor="e")
        self.granulometry_table.insert("", tk.END, values=header, tags=("header",))
        self.granulometry_table.insert("", tk.END, values=values)
        self.granulometry_table.tag_configure("header", background="lightgrey", font=(APP_FONT, 12, "bold"))
        self.root.update()

    def plot_classification(self) -> None:
        """Draw the mass-distribution pie chart and its summary table.

        The classes displayed depend on ``class_view_mode``:

        * ``"en933"`` -- the EN 933-11 super-classes (``self.EN933_preds``);
        * ``"all"``   -- every fine-grained class detected by the model
          (``self.extended_preds``).

        Both the pie chart and the summary table below it follow the selected
        mode, so the user can inspect the mass distribution of all detected
        classes, not only the aggregated EN 933-11 ones.
        """
        if self.EN933_preds is None:
            tkMessageBox.showinfo("Information", "Run a prediction first.", parent=self.root)
            return

        show_all = self.class_view_mode.get() == "all"
        summary = self.extended_preds if show_all else self.EN933_preds
        labels = list(summary["class"])
        fracs = np.asarray(summary["mass_fraction"], dtype=float)
        masses = np.asarray(summary["mass(g)"], dtype=float)
        counts = np.asarray(summary["number of elts"], dtype=int)

        # Clear the previous rendering (chart + table) before redrawing.
        self.classification_fig.clear()
        self.classification_table.delete(*self.classification_table.get_children())

        if not labels or float(masses.sum()) <= 0:
            self.classification_fig.text(0.5, 0.5, "No mass to display.", ha="center", va="center")
            self.classification_canvas.draw()
            self.root.update()
            return

        # One colour per class: super-class colours for EN 933-11, a distinct
        # colour from a qualitative colormap for every fine-grained class.
        if show_all:
            cmap = plt.get_cmap("tab20")
            n = len(labels)
            colors = [cmap(i / (n - 1)) for i in range(n)] if n > 1 else [cmap(0)]
        else:
            colors = [COLORS.get(cls, "lightgrey") for cls in labels]

        axes = self.classification_fig.add_subplot(111)
        axes.pie(
            fracs,
            labels=labels,
            autopct="%2.1f%%",
            colors=colors,
            wedgeprops={"edgecolor": "black", "linewidth": 1, "antialiased": True},
        )
        axes.set_title("Mass distribution - " + ("all detected classes" if show_all else "EN 933-11 super-classes"))
        zoom_factory(axes)
        self.classification_canvas.draw()

        # Summary table (header + fraction + mass + count rows).
        header = ["Class"] + labels
        fraction_row = ["Fraction (%)"] + [f"{f * 100:4.1f}" for f in fracs]
        mass_row = ["Mass (g)"] + [f"{m:4.2f}" for m in masses]
        count_row = ["Count"] + [str(c) for c in counts]

        self.classification_table.configure(columns=header)
        self.classification_table["show"] = ""
        for i, col in enumerate(header):
            self.classification_table.column(col, width=(100 if i == 0 else 70), stretch=False, anchor="e")
        self.classification_table.insert("", tk.END, values=header, tags=("header",))
        self.classification_table.insert("", tk.END, values=fraction_row)
        self.classification_table.insert("", tk.END, values=mass_row, tags=("alt",))
        self.classification_table.insert("", tk.END, values=count_row, tags=("alt",))
        self.classification_table.tag_configure("header", background="lightgrey", font=(APP_FONT, 12, "bold"))
        self.classification_table.tag_configure("alt", background="lightgrey")
        self.root.update()

    def plotimg(self, img) -> None:
        """Display an image array on the main image canvas (zoom/pan enabled)."""
        if img is None:
            return
        self.fig.clear()
        axes = self.fig.add_subplot(111)
        image_obj = axes.imshow(img, interpolation="none", origin="upper")
        image_obj.set_extent((0, img.shape[1], 0, img.shape[0]))
        axes.axis("off")
        self.fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        zoom_factory(axes)
        panhandler(self.fig)
        self.img_canvas.draw()
        self.root.update()

    # ------------------------------------------------------------------
    # Image I/O
    # ------------------------------------------------------------------
    def open_image(self, filename: str, plot: bool = True, lmax: int | None = DEFAULT_IMG_SIZE) -> np.ndarray | None:
        """Open an image, optionally downscale it for display, and plot it.

        Args:
            filename: Path to the image file.
            plot: If True, render the image on the main canvas.
            lmax: Maximum allowed side length (px); larger images are downscaled.

        Returns:
            The image as a numpy array, or ``None`` if it could not be opened.
        """
        if lmax is None:
            lmax = DEFAULT_IMG_SIZE
        try:
            img = Image.open(filename)
        except Exception as exc:  # noqa: BLE001 -- any read failure is shown to the user
            tkMessageBox.showerror("Error", f"Error opening {filename}:\n{exc}", parent=self.root)
            return None

        longest = max(img.size)
        if longest > lmax:
            ratio = longest / lmax
            width, height = img.size
            img = img.resize((int(width // ratio), int(height // ratio)))
        img_array = np.asarray(img)
        if plot:
            self.plotimg(img_array)
        return img_array

    # ------------------------------------------------------------------
    # Folder / list management
    # ------------------------------------------------------------------
    def init_img_list(self) -> None:
        """Scan the input folder and display the first image."""
        self.img_list = gui_utils.get_img_list(self.input_dir)
        self.img_pointer = 0
        self.reset()
        if self.img_list:
            self.current_image_fn = self.img_list[0]
            self.img = self.open_image(self._full_path(self.current_image_fn), plot=True)

    def select_input_dir(self) -> None:
        """Open a folder picker and (re)initialise the image list on selection."""
        selected = tkFileDialog.askdirectory(parent=self.root, initialdir=self.input_dir)
        if selected:
            self.input_dir = selected
            self.input_dir_var.set(self.input_dir)
            self.config_manager.config["input_folder"] = self.input_dir
            self.config_manager.save_parameters()
            self.init_img_list()

    def reset(self) -> None:
        """Reset the per-folder result state (called when the input changes)."""
        self.seg_button_text.set("View Segmentation")
        self.seg_mode = False
        self.annotations = None
        self.extended_preds = None
        self.EN933_preds = None
        self.detection_dir = None
        self.nb_aggregates_var.set("n/a")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close the application window."""
        self.root.destroy()


if __name__ == "__main__":
    app = ramsesGUI()
    app.root.mainloop()
