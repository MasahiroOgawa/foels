import os
import numpy as np
import cv2
import json
from oneformersegmentator import OneFormerSegmentator
from PIL import Image


class Segmentator:
    def __init__(
        self,
        model_name=None,
        input_dir="../data/sample",
        result_dir="result",
        thre_static_prob=0.1,
        log_level=0,
    ):
        """
        model_name: segmentation model name. Currently only used for oneformer.
        input_dir: input image directory .
        result_dir: result directory where segmentation results are stored.
        thre_static_prob: threshold of moving probability to determine static objects.
        log_level: log level. 0: no log, 1: print log, 2: display image, 3: debug with detailed image, 4: debug with detailed image and stop every step.
        """
        # constants
        self.RESULT_DIR = result_dir
        self.INPUT_DIR = input_dir
        self.THRE_STATIC_PROB = thre_static_prob
        self.LOG_LEVEL = log_level
        self.THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        self.ID_CLASS_FILENAME = "id_class.json"
        self.CLASS_MOVPROB_FILENAME = "class_prob.json"

        # variables
        self.model_name = model_name
        self.result_img = None
        self.result_mask = None
        self.moving_prob = None
        self.moving_prob_img = None
        self.result_movingobj_img = None
        self.class_movprobs = None
        self.sky_id = None
        self.static_ids = []
        self.sky_mask = None
        self.static_mask = None

    def load_prior(self):
        self._load_class_movprobs()
        self._comp_sky_id()
        self._comp_static_ids()

    def compute(self, img_name):
        pass

    def _comp_moving_prob(self):
        self.moving_prob = np.zeros_like(self.result_mask, dtype=float)
        for row in range(self.result_mask.shape[0]):
            for col in range(self.result_mask.shape[1]):
                class_id = self.result_mask[row, col]
                self.moving_prob[row, col] = self.class_movprobs[class_id][
                    "moving_prob"
                ]

    def draw(self, bg_img=None):
        self.bg_img = bg_img
        self._draw_movingobj_img()
        self._draw_movingprob_img()

    def _draw_movingobj_img(self):
        self.result_movingmask_img = self.bg_img.copy() // 2

        # draw sky mask as light blue in result_movingobj_img
        self.result_movingmask_img[self.sky_mask, 0] += 128  # 0 means blue channel

        # draw moving object mask as blighter in result_movingobj_img
        moving_obj_mask = np.logical_not(np.logical_or(self.sky_mask, self.static_mask))
        self.result_movingmask_img[moving_obj_mask, :] += 128
        cv2.putText(
            self.result_movingmask_img,
            "moving mask",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

    def _draw_movingprob_img(self):
        # draw moving probability as jet color in moving_prob_img.
        self.moving_prob_img = np.zeros(
            (self.moving_prob.shape[0], self.moving_prob.shape[1], 3), dtype=np.uint8
        )

        self.moving_prob_img = cv2.applyColorMap(
            (self.moving_prob * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        cv2.putText(
            self.moving_prob_img,
            "prior",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

    def _load_class_movprobs(self):
        class_movprob_fname = os.path.join(
            self.THIS_DIR, "..", "data", self.CLASS_MOVPROB_FILENAME
        )
        if not os.path.exists(class_movprob_fname):
            self.dump_classes_with_moving_prob()

        self.class_movprobs = json.load(open(class_movprob_fname, "r"))

    def dump_classes_with_moving_prob(self):
        """
        name: dump classes with moving probability.
        usage: Just use this function when you need to change the class names in the class file.
        """
        if self.class_movprobs is None:
            if (
                self.model_name == "shi-labs/oneformer_coco_swin_large"
            ):  # oneformer case
                self.dump_id_class_json()
            else:
                print("[ERROR] id-class file is not exist. Please create it first")
                exit()

        self.id_classes = json.load(
            open(os.path.join(self.RESULT_DIR, self.ID_CLASS_FILENAME), "r")
        )
        # convert id from string to int, because id needs to compare as an int later.
        self.id_classes = {int(k): v for k, v in self.id_classes.items()}

        # create combined struct with id, class name and moving probagilities, which is 0 as default.
        self.class_movprobs = []
        # append "moving_prob": 0.0 for each content.
        for id, class_name in self.id_classes.items():
            self.class_movprobs.append(
                {"class_name": class_name, "class_id": id, "moving_prob": 0.01}
            )

        # save classes with mobing probability
        self.class_movprob_file = os.path.join(
            self.THIS_DIR, "..", "data", self.CLASS_MOVPROB_FILENAME
        )
        json.dump(self.class_movprobs, open(self.class_movprob_file, "w"))

    def _comp_sky_id(self):
        for class_dict in self.class_movprobs:
            if class_dict["class_name"] in {"sky", "sky-other-merged"}:
                self.sky_id = class_dict["class_id"]
                break

    def _comp_sky_mask(self):
        if self.sky_id is not None:
            self.sky_mask = self.result_mask == self.sky_id
        else:
            self.sky_mask = np.zeros_like(self.result_mask, dtype=bool)

    def _comp_static_ids(self):
        for class_dict in self.class_movprobs:
            if (class_dict["moving_prob"] < self.THRE_STATIC_PROB) and (
                class_dict["class_id"] != self.sky_id
            ):
                self.static_ids.append(class_dict["class_id"])

    def _comp_static_mask(self):
        if len(self.static_ids) > 0:
            self.static_mask = np.zeros_like(self.result_mask, dtype=bool)
            for static_id in self.static_ids:
                self.static_mask = np.logical_or(
                    self.static_mask, (self.result_mask == static_id)
                )


class InternImageSegmentatorWrapper(Segmentator):
    # currenly just load the result from already processd direwhere input images are stored.ctory.
    def __init__(
        self,
        model_name=None,
        input_dir=None,
        result_dir="result",
        thre_static_prob=0.1,
        log_level=0,
    ):
        """
        model_name: InternImageSegmentatorWrapper just load the result from already processed directory.
                    So this model_name is not used.
        """
        super().__init__(model_name, input_dir, result_dir, thre_static_prob, log_level)

    def compute(self, img_name):
        if self.LOG_LEVEL > 0:
            print(f"[INFO] InternImageSegmentator.compute({img_name})")

        # get segmentation image
        seg_imgfile = os.path.join(self.RESULT_DIR, img_name)
        self.result_img = cv2.imread(seg_imgfile)

        # get segmentation result
        imgnum = img_name.split(".")[0]
        seg_resultfile = os.path.join(self.RESULT_DIR, f"{imgnum}.npy")
        self.result_mask = np.load(seg_resultfile)

        self._comp_sky_mask()
        self._comp_static_mask()
        self._comp_moving_prob()


class OneFormerSegmentatorWrapper(Segmentator):
    def __init__(
        self,
        model_name="shi-labs/oneformer_coco_swin_large",
        task_type="panoptic",
        input_dir="../data/sample",
        result_dir="result",
        thre_static_prob=0.1,
        log_level=0,
    ):
        super().__init__(model_name, input_dir, result_dir, thre_static_prob, log_level)
        self.oneformer = OneFormerSegmentator(model_name, task_type)

    def compute(self, img_name):
        if self.LOG_LEVEL > 0:
            print(f"[INFO] OneformerSegmentator.compute({img_name})")

        # load PIL image
        image = Image.open(os.path.join(self.INPUT_DIR, img_name))

        # run segmentation
        # result_mask: segmentation result as class id, so semantic segmentation.
        # id_mask: segmentation result as instance id, so instance segmentation.
        self.result_mask, self.id_mask, segments_info = self.oneformer.inference(image)
        self.result_img = self.oneformer.result_cvmat()

        if self.LOG_LEVEL > 0:
            self.oneformer.print_result()

        self._comp_sky_mask()
        self._comp_static_mask()
        self._comp_moving_prob()

    def dump_id_class_json(self):
        id2label = self.oneformer.model.config.id2label
        json.dump(
            id2label, open(os.path.join(self.RESULT_DIR, self.ID_CLASS_FILENAME), "w")
        )

    def show(self):
        self.oneformer.show()
