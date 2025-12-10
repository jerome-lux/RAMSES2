import cv2
import numpy as np

from itertools import cycle
from skimage.color import label2rgb, rgb2gray
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops_table

from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import TextArea, AnnotationBbox
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd

_CLASS_COLORS = {
    "Ra": "dimgrey",
    "Rb01": "darkorange",
    "Rb02": "sandybrown",
    "Rc": "lightblue",
    "Rcu01": "yellow",
    "Ru01": "ivory",
    "Ru02": "slateblue",
    "Ru03": "lightgrey",
    "Ru04": "peru",
    "Ru05": "pink",
    "Ru06": "slategrey",
    "Rg": "limegreen",
    "X01": "burlywood",
    "X02": "lightblue",
    "X03": "red",
    "X04": "tan",
    "UNKNOWN": "violet",
    "Coin": "gold",
}

_EN93311_COLORS = {
    "Ra": "dimgrey",
    "Rb": "darkorange",
    "Rc": "lightblue",
    "Rg": "limegreen",
    "Ru": "ivory",
    "X": "red",
    "UNKNOWN": "violet",
    "Coin": "gold",
}

_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.143,
            0.143,
            0.143,
            0.857,
            0.857,
            0.857,
            1.000,
            1.000,
            1.000,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)


def draw_instances(
    image,
    labeled_masks,
    cls_ids,
    cls_scores=None,
    bboxes=None,
    draw_boundaries=True,
    showtext=True,
    drawrect=False,
    mode="instance",
    colors=None,
    fontcolor=(0, 0, 0),
    alpha=0.5,
    thickness=2,
    fontscale=1,
    fontface=cv2.FONT_HERSHEY_DUPLEX,
    padding=5,
    boundary_mode="inner",
):
    """draw masks with class labels and probabilities using opencv
    inputs:
    image [H, W, 3]: input image to draw boxes onto
    labeled_masks [H/2, W/2] : labeled masks
    cls_ids [N]: tensor of class ids
    cls_probs [N]: tensor of class scores
    class_ids_to_name: a dict mapping cls_ids to their names
    show: plot the image
    fontscale : for class names
    alpha: control the labels transparency
    returns:
        annotated image (RGB image as uint8 np.array)

    """
    assert mode in ["instance", "class"], "mode must be either 'instance' or 'class' "

    # Set colormap
    if mode == "class" and cls_ids is None:
        print("You must provide the class ids when mode == 'class'. Switching to instance mode")
        mode = "instance"

    if colors is None:
        if mode == "class":
            colors = [[0, 0, 0]]  # add for bg
            colors.extend([mpl.colors.to_rgb(_CLASS_COLORS[key]) for key in cls_ids])
            colors = 255 * np.array(colors)

        else:
            colors = np.zeros((_COLORS.shape[0] + 1, 3))
            colors[1:,] = 255 * _COLORS

    # Convert to BGR
    colors = colors[:, ::-1].astype(np.uint8)

    # Resize
    nx, ny = labeled_masks.shape[:2]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (ny, nx), interpolation=cv2.INTER_LINEAR) * 255
    image = image.astype(np.uint8)

    # Draw overlay
    color_overlay = colors[labeled_masks].astype(np.uint8)
    output_image = image.copy()
    mask_bool = labeled_masks > 0

    output_image[mask_bool] = cv2.addWeighted(image[mask_bool], 1 - alpha, color_overlay[mask_bool], alpha, 0)

    # Draw Boundaries
    if draw_boundaries:
        bd = find_boundaries(labeled_masks, connectivity=2, mode=boundary_mode, background=0).astype(np.uint8)
        boundary_mask = bd > 0
        # Définir ces pixels à la couleur noire [0, 0, 0] dans l'output_image
        output_image[boundary_mask] = [0, 0, 0]

    if showtext:
        # First compute all bouding boxes
        if bboxes is None:
            props_table = regionprops_table(labeled_masks, properties=["label", "bbox"])
            df = pd.DataFrame(props_table)
            labels = df["label"].values
            bboxes = df[["bbox-0", "bbox-1", "bbox-2", "bbox-3"]].values
        else:
            props_table = regionprops_table(labeled_masks, properties=["label"])
            df = pd.DataFrame(props_table)
            labels = df["label"].values

        # Place text
        for i in range(len(bboxes)):
            ymin, xmin, ymax, xmax = bboxes[i]
            lab = labels[i]

            # Préparation du texte
            if cls_scores is None:
                classtext = f"{cls_ids[lab - 1]}"
            else:
                classtext = f"{cls_ids[lab - 1]}:{cls_scores[lab - 1 ] * 100:.0f}%"

            current_color = [int(c) for c in colors[lab]]

            # Calcul de la taille du texte
            (text_w, text_h), baseline = cv2.getTextSize(classtext, fontface, fontscale, thickness)

            # positionnement
            # Tente de placer le texte AU-DESSUS (valeur Y du bas du fond)
            y_text_bottom = ymin - padding

            # Calcul Y de la ligne du haut du fond du texte
            y_text_top = y_text_bottom - text_h - baseline - padding

            # Vérification si le texte sort par le haut (y < 0)
            if y_text_top < 0:
                # Place EN-DESSOUS
                y_text_top = ymax + padding
                y_text_bottom = y_text_top + text_h + baseline + padding

                # Si ça sort du bas de l'image, on force l'affichage en haut de l'objet
                if y_text_bottom > nx:
                    y_text_top = ymin + padding
                    y_text_bottom = y_text_top + text_h + baseline + padding

            # Coordonnées du coin supérieur gauche du texte (pour l'alignement)
            # Aligné à gauche de la boîte englobante
            x_text_start = xmin

            # Ajustement pour que le texte ne sorte pas à droite de l'image
            if x_text_start + text_w + 2 * padding > ny:
                x_text_start = ny - text_w - 2 * padding

            if drawrect:
                cv2.rectangle(
                    output_image,
                    (x_text_start - padding, y_text_top),
                    (x_text_start + text_w + padding, y_text_bottom),
                    current_color,
                    thickness=cv2.FILLED,
                )

            cv2.putText(
                output_image,
                classtext,
                org=(x_text_start, y_text_bottom - baseline),
                fontFace=fontface,
                fontScale=fontscale,
                color=fontcolor,
                thickness=thickness,
            )

    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)


def plot_instances(
    image,
    mask,
    cls_ids=None,
    cls_scores=None,
    mode="instance",
    alpha=0.25,
    box_alpha=0.75,
    fontsize=8,
    fontcolor="black",
    fontweight="normal",
    draw_boundaries=True,
    dpi=400,
    boundary_mode="thick",
    show=False,
    x_offset=20,
    y_offset=20,
):
    """Draw predicted masks onto image, with associated predicted class
    returns a matplotlib figure"""

    assert mode in ["instance", "class"], "mode must be either 'instance' or 'class' "

    if mode == "class" and cls_ids is None:
        print("You must provide the class ids when mode == 'class'. Switching to instance mode")
        mode = "instance"

    nx, ny = mask.shape[:2]
    labels = np.unique(mask)
    image = cv2.resize(image, (ny, nx), interpolation=cv2.INTER_LINEAR)

    fig, ax = plt.subplots(1, 1, figsize=(ny / dpi, nx / dpi), dpi=dpi)

    if draw_boundaries:
        bd = find_boundaries(mask, connectivity=2, mode=boundary_mode, background=0).astype(np.uint8)
        bd = np.ma.masked_equal(bd, 0)

    ax.imshow(image)  # , extent = (0, image.shape[1], 0, image.shape[0]))
    masked_mask = np.ma.masked_equal(mask, 0)

    if mode == "instance":
        cmap = ListedColormap(_COLORS, N=labels.size - 1)
        ax.imshow(
            masked_mask, cmap=cmap, interpolation="nearest", alpha=alpha
        )  # , extent = (0, image.shape[1], 0, image.shape[0]))
    elif mode == "class":
        colors = [mpl.colors.to_rgb(_CLASS_COLORS[key]) for key in cls_ids]
        cmap = ListedColormap(colors, N=labels.size - 1)

    if cls_ids is not None and cls_scores is not None:
        for i, label in enumerate(labels[1:]):
            # for i, (cls_id, score, label) in enumerate(zip(cls_ids, cls_scores, labels[1:])):

            current_color = cmap.colors[i]

            # txt_cls = f"{cls_id}:{score:.2f}"
            txt_cls = f"{cls_ids[label - 1]}:{cls_scores[label - 1]:.2f}"

            coords = np.nonzero(mask == label)
            # Here there can be outliers points so we use the mean coordinates to define the location of the textbox
            # yc = max((np.mean(coords[0]) + np.min(coords[0])) // 2, 0)
            yc = max(np.mean(coords[0]) - y_offset, 0)
            xc = max(np.mean(coords[1]) - x_offset, 0)

            ax.annotate(
                txt_cls,
                xy=(xc, yc),
                xycoords="data",
                fontsize=fontsize,
                weight=fontweight,
                color=fontcolor,
                bbox=dict(fc=current_color, ec="none", alpha=box_alpha, boxstyle="square,pad=0.15"),
                horizontalalignment="left",
                verticalalignment="top",
            )

    if draw_boundaries:
        ax.imshow(bd, cmap=ListedColormap([[0, 0, 0]], N=2), alpha=1)

    ax.axis("off")
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    if show:
        plt.show()

    return fig
