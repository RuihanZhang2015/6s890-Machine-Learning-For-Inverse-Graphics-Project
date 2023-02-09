# Visualization related utility functions


def torchImageToPlottable(image):
    """
    Convert torch-conventioned image to matplot plot-able image
    @param image: [1xCxHxW Tensor] Torch-conventioned image
    @return image_plt: [HxWx1 numpy] Plot-able numpy image
    """
    assert len(image.shape) in [3, 4] and image.shape[-3] in [1, 3], \
        "Error: Image size must be (*, 1 or 3, H, W)"
    assert len(image.shape) != 4 or image.shape[0] == 1, \
        "Error: Image size must be (1, *, *, *)"
    # Squeeze the batch dimension if any
    if len(image.shape) == 4:
        image = image.squeeze(0)
    img_plt = image.permute(1, 2, 0).detach().cpu().numpy()
    return img_plt
