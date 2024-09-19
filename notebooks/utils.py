import numpy as np
import copick


def project_tomogram(vol: np.ndarray, zSlice: int = None, deltaZ: int = None):
    """
    Projects a tomogram along the z-axis.

    Parameters:
    vol (np.ndarray): 3D tomogram array (z, y, x).
    zSlice (int, optional): Specific z-slice to project. If None, project along all z slices.
    deltaZ (int, optional): Thickness of slices to project. Used only if zSlice is specified. If None, project a single slice.

    Returns:
    np.ndarray: 2D projected tomogram.
    """

    if zSlice is not None:
        # If deltaZ is specified, project over zSlice to zSlice + deltaZ
        if deltaZ is not None:
            zStart = max(zSlice, 0)
            zEnd = min(zSlice + deltaZ, vol.shape[0])  # Ensure we don't exceed the volume size
            projection = np.sum(vol[zStart:zEnd,], axis=0)  # Sum over the specified slices
        else:
            # If deltaZ is not specified, project just a single z slice
            projection = vol[zSlice,]
    else:
        # If zSlice is None, project over the entire z-axis
        projection = np.sum(vol, axis=0)

    # test_data_norm = (test_data - test_data.min()) / (test_data.max() - test_data.min())
    # image = np.repeat(test_data_norm[..., None], 3, axis=2)

    return projection


def get_coordinates(run, name="lysosome", voxel_size=10):
    picks = run.get_picks(object_name="lysosome")
    if len(picks) == 0:
        print(f"Warning: No picks found for the {name} in run {run.name}.")
        return None
    points = picks[0].points
    coordinates = np.zeros([
        len(points),
        3,
    ])  # Create an empty array to hold the (z, y, x) coordinates

    # Iterate over all points and convert their locations to coordinates in voxel space
    for ii in range(len(points)):
        coordinates[ii,] = [
            points[ii].location.z / voxel_size,  # Scale z-coordinate by voxel size
            points[ii].location.y / voxel_size,  # Scale y-coordinate by voxel size
            points[ii].location.x / voxel_size,
        ]  # Scale x-coordinate by voxel size
    points = run.get_picks(object_name=name)[0].points

    return coordinates


def get_tomogram(run, voxel_size=10, algorithm="denoised"):
    print(
        f"Getting {algorithm} Tomogram with {voxel_size} A voxel size for the associated runID: {run.name}"
    )
    tomogram = run.get_voxel_spacing(voxel_size)
    if tomogram is not None:
        tomogram = tomogram.get_tomogram(algorithm).numpy()
    return tomogram


def check_tomogram(run, voxel_size=10, algorithm="denoised"):
    tomogram = run.get_voxel_spacing(voxel_size)
    return True if tomogram is not None else False


def prepare_img_for_mask_generation(img: np.ndarray):
    im_norm = (img - img.min()) / (img.max() - img.min())
    im_norm = im_norm * 2 - 1
    img_new = np.repeat(im_norm[..., None], 3, axis=2)
    return img_new


def show_anns(ax, anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax.set_autoscale_on(False)

    img = np.ones((
        sorted_anns[0]["segmentation"].shape[0],
        sorted_anns[0]["segmentation"].shape[1],
        4,
    ))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2

            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Try to smooth contours
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)
