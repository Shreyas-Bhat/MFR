from PIL import Image
import numpy as np
import torch

from misc_functions import get_example_params, save_class_activation_images


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


class LayerCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = guided_gradients
        weights[weights < 0] = 0 # discard negative gradients
        # Element-wise multiply the weight with its conv output and then, sum
        cam = np.sum(weights * target, axis=0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255

        return cam


from PIL import Image
import cv2
import numpy as np
import torch.nn.functional as F

from model.xcos_modules import l2normalize


class GradientExtractor:
    """ Extracting activations and
    registering gradients from targetted intermediate layers
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, img1, img2):
        self.gradients = []
        feat1, out1 = self.forward(img1)
        feat2, out2 = self.forward(img2)
        return feat1, feat2, out1, out2

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def forward(self, x):
        feats = []
        x = self.model.input_layer(x)
        x = self.model.body(x)
        x.register_hook(self.save_gradient)
        feats.append(x)
        x = self.model.output_layer(x)
        return feats, x


class ModelOutputs:
    """ Making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers.
    """

    def __init__(self, model):
        self.model = model
        self.extractor = GradientExtractor(self.model)

    def get_grads(self):
        return self.extractor.gradients

    def __call__(self, img1, img2):
        feat1, feat2, out1, out2 = self.extractor(img1, img2)

        out1 = l2normalize(out1)
        out2 = l2normalize(out2)
        cos = F.cosine_similarity(out1, out2, dim=1, eps=1e-6)

        return feat1, feat2, cos


class FaceLayerCam:

    def __init__(self, model):
        self.model = model
        self.extractor = ModelOutputs(self.model)

    def __call__(self, img1, img2):
        feat1, feat2, output = self.extractor(img1, img2)

        self.model.zero_grad()
        output.backward(retain_graph=True)
        grads = self.extractor.get_grads()
        hm1 = self.make_heatmap(grads[0].cpu().data.numpy(), feat1[0].cpu().data.numpy())
        hm2 = self.make_heatmap(grads[1].cpu().data.numpy(), feat2[0].cpu().data.numpy())

        return hm1, hm2

    def make_heatmap(self, grad, feat):
        """Batch operation supported
        """
        weights = np.mean(grad, axis=(-2, -1), keepdims=True)
        x = weights * feat

        x = x.sum(axis=1)
        x = np.maximum(0, x)
        x = x - np.min(x, axis=(-2, -1), keepdims=True)
        x = x / np.max(x, axis=(-2, -1), keepdims=True)
        x = 1. - x
        return x

    def make_img(self, heatmap, size, ori_img=None):
        """Batch operation NOT suppored
        """
        hm = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        hm = cv2.resize(hm, size)
        if ori_img is not None:
            hm = np.float32(hm) / 255 + np.transpose(ori_img.numpy(), (1, 2, 0)) * 0.5 + 0.5
            hm /= np.max(hm)
            hm = np.uint8(255 * hm)
        return Image.fromarray(hm)
