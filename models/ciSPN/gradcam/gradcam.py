import numpy as np
import torch
from scipy.ndimage.interpolation import zoom


class CamExtractor:
    """
    Extracts cam features from the model
    """

    def __init__(self, model):
        self.model = model
        self.cnn_gradients = None

    def save_cnn_gradient(self, grad):
        self.cnn_gradients = grad

    def forward_pass(self, input_image, target_shape):
        self.conv_output = None

        def cnn_layer_hook(module, input, output):
            # register hook for gradients and save conv output
            output.register_hook(self.save_cnn_gradient)
            self.conv_output = output

        # we assume that the first sequential is a cnn block
        first_block = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Sequential):
                if first_block is None:
                    first_block = module
        first_block.register_forward_hook(cnn_layer_hook)

        target_placeholder = torch.zeros(target_shape, dtype=torch.float, device="cuda")
        marginalized = torch.ones(target_shape, dtype=torch.float, device="cuda")
        y = self.model.predict(input_image, target_placeholder, marginalized)
        return self.conv_output, y


class GradCam:
    """
    Produces class activation map
    """

    def __init__(self, model):
        self.model = model
        self.extractor = CamExtractor(self.model)

    def generate_cam(self, input_image, target):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)

        self.model.eval()
        conv_output, model_output = self.extractor.forward_pass(
            input_image, target.shape
        )

        self.model.zero_grad()  # Zero grads

        # gradient w.r.t. to expected target:
        # one_hot_output = [0, 0, 1, 0, 0]
        model_output.backward(gradient=target, retain_graph=True)

        # Get hooked gradients
        guided_gradients = self.extractor.cnn_gradients.data.cpu().numpy()
        guided_gradients = guided_gradients[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        # Get weights from gradients
        weights = np.mean(
            guided_gradients, axis=(1, 2)
        )  # take average gradient over channels for each pixel
        weights = np.abs(weights)  # consider magnitude and not the sign of gradients

        cam = np.zeros(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        mn = np.min(cam)
        mx = np.max(cam)
        cam = (cam - mn) / (mx - mn)  # Normalize between 0-1

        cam = zoom(cam, np.array(input_image[0].shape[1:]) / np.array(cam.shape))
        return cam
