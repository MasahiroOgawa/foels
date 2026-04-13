import sys
import os
import numpy as np
import torch
import logging
from typing import Optional, Tuple
import numpy.typing as npt
from ext.unimatch.utils import frame_utils, flow_viz
from PIL import Image

# Setup logger
logger = logging.getLogger(__name__)

# Constants for normalization
PIXEL_NORMALIZATION_SCALE = 255.0


class UnimatchFlow:
    """
    compute optical flow using unimatch algorithm
    """

    def __init__(self, FLOW_RESULT_DIR) -> None:
        self.FLOW_RESULT_DIR = FLOW_RESULT_DIR

    def compute(self, imgname):
        """
        compute optical flow from 2 consecutive images.
        currently just read flow from files.
        args:
            imgname: image file name. e.g. 00000.jpg
        result:
            self.flow: size = h x w x 2. 2 means flow vector (u,v).
            self.flow_img: size = h x w x 3. 3 means RGB channel which represents flow orientation.
        """
        imgnum = imgname.split(".")[0]
        flow_file = os.path.join(self.FLOW_RESULT_DIR, f"{imgnum}_pred.flo")
        if flow_file is None:
            raise ValueError(f"flow file {flow_file} does not exist.")
        self.flow = frame_utils.readFlow(flow_file)

        self.flow_img = flow_viz.flow_to_image(self.flow)


class MemFlow:
    """
    Compute optical flow using MemFlow algorithm
    """

    def __init__(
        self,
        model_name: str = "MemFlowNet",
        stage: str = "things",
        weights_path: Optional[str] = None,
    ) -> None:
        """
        Initialize MemFlow model
        args:
            model_name: MemFlowNet or MemFlowNet_T
            stage: things, sintel, kitti, spring
            weights_path: path to pretrained weights
        """
        # Validate memflow submodule exists
        memflow_path = os.path.join(os.path.dirname(__file__), "ext/memflow")
        if not os.path.exists(memflow_path):
            raise RuntimeError(
                "MemFlow submodule not found. Please run: "
                "git submodule update --init --recursive"
            )

        # Add to path with error handling
        if memflow_path not in sys.path:
            sys.path.append(memflow_path)
            sys.path.append(os.path.join(memflow_path, "core"))

        # Import MemFlow components with proper error handling
        try:
            if stage == "things":
                from configs.things_memflownet import get_cfg
            elif stage == "sintel":
                from configs.sintel_memflownet import get_cfg
            elif stage == "kitti":
                from configs.kitti_memflownet import get_cfg
            elif stage == "spring":
                from configs.spring_memflownet import get_cfg
            else:
                raise ValueError(
                    f"Unknown stage: {stage}. Must be one of: things, sintel, kitti, spring"
                )
        except ImportError as e:
            raise ImportError(
                f"Failed to import MemFlow config for stage '{stage}'. "
                f"Ensure MemFlow dependencies are installed: {e}"
            )

        try:
            from core.Networks import build_network
            from utils.utils import InputPadder
            from utils import flow_viz as memflow_viz
            from inference import inference_core_skflow as inference_core
        except ImportError as e:
            raise ImportError(
                f"Failed to import MemFlow modules. "
                f"Ensure all dependencies are installed: {e}"
            )

        # Store imports for later use
        self.InputPadder = InputPadder
        self.memflow_viz = memflow_viz
        self.inference_core = inference_core

        # Get configuration
        self.cfg = get_cfg()
        self.cfg.restore_ckpt = weights_path

        # Check if CUDA is available with proper error handling
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device.type == "cpu":
                logger.warning(
                    "CUDA not available, using CPU. Performance will be slower."
                )
        except Exception as e:
            logger.error(f"Error setting up compute device: {e}")
            self.device = torch.device("cpu")

        # Create model
        logger.info(f"Creating {model_name} model on {self.device}...")
        try:
            self.model = build_network(self.cfg).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to create MemFlow model: {e}")

        # Load pretrained weights if available
        if weights_path:
            if not os.path.exists(weights_path):
                logger.warning(
                    f"Pretrained model not found at {weights_path}. Using random initialization."
                )
            else:
                try:
                    logger.info(f"Loading pretrained model from {weights_path}")
                    checkpoint = torch.load(weights_path, map_location=self.device)

                    # Handle different checkpoint formats
                    if "model" in checkpoint:
                        state_dict = checkpoint["model"]
                    else:
                        state_dict = checkpoint

                    # Remove 'module.' prefix if present
                    if state_dict and "module" in list(state_dict.keys())[0]:
                        new_state_dict = {}
                        for key in state_dict.keys():
                            new_state_dict[key.replace("module.", "", 1)] = state_dict[
                                key
                            ]
                        state_dict = new_state_dict

                    self.model.load_state_dict(state_dict, strict=False)
                    logger.info("MemFlow model loaded successfully!")
                except Exception as e:
                    logger.error(f"Failed to load pretrained weights: {e}")
                    raise RuntimeError(
                        f"Could not load model weights from {weights_path}: {e}"
                    )

        self.model.eval()

        # Create processor for inference
        self.processor = self.inference_core.InferenceCore(self.model, config=self.cfg)

        # Initialize flow storage
        self.flow = None
        self.flow_img = None

    def _compute_flow(
        self, image1_tensor: torch.Tensor, image2_tensor: torch.Tensor
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        """
        Internal method to compute flow from preprocessed tensors.
        args:
            image1_tensor: First image tensor (C x H x W)
            image2_tensor: Second image tensor (C x H x W)
        returns:
            flow: Flow array (H x W x 2)
            flow_img: Flow visualization (H x W x 3)
        """
        try:
            # Stack images
            images = torch.stack([image1_tensor, image2_tensor]).to(self.device)

            # Pad images for inference
            padder = self.InputPadder(images.shape)
            images = padder.pad(images)

            # Normalize pixel values from [0, 255] to [-1, 1] range expected by model
            images = 2 * (images / PIXEL_NORMALIZATION_SCALE) - 1.0

            # Run inference
            with torch.no_grad():
                # Process image pair (batch_size=1, time=2)
                images_batch = images.unsqueeze(0)  # Add batch dimension
                flow_low, flow_pred = self.processor.step(
                    images_batch, end=True, add_pe=False
                )
                flow = padder.unpad(flow_pred[0]).cpu().numpy()

            # Convert to H x W x 2 format (transpose from 2 x H x W)
            flow = np.transpose(flow, (1, 2, 0))
            flow_img = self.memflow_viz.flow_to_image(flow)

            return flow, flow_img
        except Exception as e:
            logger.error(f"Error computing optical flow: {e}")
            raise RuntimeError(f"Failed to compute optical flow: {e}")

    def compute_from_images(
        self, img1_path: str, img2_path: str
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        """
        Compute optical flow from two image paths
        args:
            img1_path: path to first image
            img2_path: path to second image
        returns:
            flow: size = H x W x 2. 2 means flow vector (u,v).
            flow_img: size = H x W x 3. 3 means RGB channel which represents flow orientation.
        """
        # Validate input paths
        for path in [img1_path, img2_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image file not found: {path}")

        try:
            # Load images
            img1 = np.array(Image.open(img1_path)).astype(np.uint8)
            img2 = np.array(Image.open(img2_path)).astype(np.uint8)
            return self.compute_from_arrays(img1, img2)
        except Exception as e:
            logger.error(f"Error loading images: {e}")
            raise RuntimeError(f"Failed to load images for flow computation: {e}")

    def compute_from_arrays(
        self, img1_array: npt.NDArray[np.uint8], img2_array: npt.NDArray[np.uint8]
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        """
        Compute optical flow from two numpy arrays
        args:
            img1_array: first image as numpy array (H x W x 3)
            img2_array: second image as numpy array (H x W x 3)
        returns:
            flow: size = H x W x 2. 2 means flow vector (u,v).
            flow_img: size = H x W x 3. 3 means RGB channel which represents flow orientation.
        """
        # Validate input shapes
        if img1_array.shape != img2_array.shape:
            raise ValueError(
                f"Image shapes must match. Got {img1_array.shape} and {img2_array.shape}"
            )
        if len(img1_array.shape) != 3 or img1_array.shape[2] != 3:
            raise ValueError(f"Images must be H x W x 3. Got shape {img1_array.shape}")

        try:
            # Convert to tensors
            img1 = torch.from_numpy(img1_array).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2_array).permute(2, 0, 1).float()

            # Compute flow using internal method
            self.flow, self.flow_img = self._compute_flow(img1, img2)
            return self.flow, self.flow_img
        except Exception as e:
            logger.error(f"Error processing image arrays: {e}")
            raise RuntimeError(f"Failed to compute flow from arrays: {e}")


class UndominantFlowAngleExtractor:
    def __init__(
        self, THRE_FLOWLENGTH=4.0, THRE_ANGLE=10 * np.pi / 180, LOG_LEVEL=0
    ) -> None:
        # constants
        # if flow length is lower than this value, the flow is ignored.
        self.THRE_FLOWLENGTH = THRE_FLOWLENGTH
        self.LOG_LEVEL = LOG_LEVEL
        self.THRE_ANGLE = THRE_ANGLE  # radian

    def compute(self, flow: np.ndarray, nonsky_static_mask: np.ndarray):
        """
        compute undominant orientation mask from optical flow.
        args:
            flow: size = h x w x 2. 2 means flow vector (u,v).
        result:
            self.undominant_flow_prob: size = h x w.
        """
        # compute flow angle and length
        flow_angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])
        flow_length = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

        # extract median angle
        median_angle = np.median(flow_angle[nonsky_static_mask])

        # compute mask from median angle.
        self.undominant_flow_prob = np.zeros(
            (flow.shape[0], flow.shape[1]), dtype=np.float16
        )

        self.undominant_flow_prob[
            (flow_length > self.THRE_FLOWLENGTH)
            & (np.abs(flow_angle - median_angle) > self.THRE_ANGLE)
        ] = 0.9  # 0.9 means outlier
        self.undominant_flow_prob[
            (flow_length > self.THRE_FLOWLENGTH)
            & (np.abs(flow_angle - median_angle) <= self.THRE_ANGLE)
        ] = 0.1  # 0.1 means inlier
