import cv2
import numpy as np

from itertools import cycle
from skimage.color import label2rgb, rgb2gray
from skimage.segmentation import find_boundaries

from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import TextArea, AnnotationBbox
import matplotlib.pyplot as plt


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
    class_ids_to_name=None,
    draw_boundaries=True,
    show=True,
    colors=None,
    fontscale=5,
    fontcolor=(0, 0, 0),
    alpha=0.5,
    thickness=1,
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
    if colors is None:
        colors = _COLORS

    nx, ny = labeled_masks.shape

    img = cv2.resize(image, (ny, nx), interpolation=cv2.INTER_LINEAR)

    output_image = label2rgb(labeled_masks, img, bg_label=0, alpha=alpha, colors=colors)

    if draw_boundaries:
        bd = find_boundaries(labeled_masks, connectivity=2, mode="inner", background=0)
        output_image = np.where(bd[..., np.newaxis], (0, 0, 0), output_image)

    if cls_scores is None:
        cls_scores = np.ones(cls_ids.size)

    # Show class names and scores
    if class_ids_to_name is not None and fontscale > 0:

        colors = cycle(colors)

        for i, (class_id, class_score) in enumerate(zip(cls_ids, cls_scores)):

            current_color = next(colors).tolist()

            coords = np.where(labeled_masks == i + 1)

            yc = np.mean(coords[:, 0])
            xc = np.mean(coords[:, 1])

            class_name = class_ids_to_name[class_id]
            classtext = "{}:{:.0f}%".format(class_name, class_score * 100)
            (text_width, text_height), baseline = cv2.getTextSize(
                classtext, cv2.FONT_HERSHEY_SIMPLEX, fontscale, thickness
            )
            ymin_txt = yc - baseline if yc - baseline - text_height > 0 else yc + text_height
            ymin_bg = yc - text_height - baseline if yc - text_height - baseline > 0 else yc + text_height + baseline
            cv2.rectangle(
                output_image,
                (xc - text_width // 2, ymin_bg),
                (xc + text_width // 2, yc),
                current_color,
                thickness=cv2.FILLED,
            )
            cv2.putText(
                output_image,
                classtext,
                org=(xc - text_width // 2, ymin_txt),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=fontscale,
                color=fontcolor,
                thickness=thickness,
            )

    if show:
        plt.imshow(output_image)
        plt.show()
        return output_image
    else:
        return np.round(output_image * 255).astype(np.uint8)


def plot_instances(
    image,
    mask,
    cls_ids=None,
    cls_scores=None,
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

    nx, ny = mask.shape[:2]
    labels = np.unique(mask)
    image = cv2.resize(image, (ny, nx), interpolation=cv2.INTER_LINEAR)

    fig, ax = plt.subplots(1, 1, figsize=(ny / dpi, nx / dpi), dpi=dpi)

    if draw_boundaries:
        bd = find_boundaries(mask, connectivity=2, mode=boundary_mode, background=0).astype(np.uint8)
        bd = np.ma.masked_equal(bd, 0)

    ax.imshow(image)  # , extent = (0, image.shape[1], 0, image.shape[0]))
    masked_mask = np.ma.masked_equal(mask, 0)

    cmap = ListedColormap(_COLORS, N=labels.size)
    ax.imshow(
        masked_mask, cmap=cmap, interpolation="nearest", alpha=alpha
    )  # , extent = (0, image.shape[1], 0, image.shape[0]))

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
