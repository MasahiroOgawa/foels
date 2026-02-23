import logging
from enum import Enum

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
CameraState = Enum("CameraState", ["STOPPING", "ROTATING", "ONLY_TRANSLATING"])


class FoE:
    def __init__(
        self,
        thre_flowlength=1.0,
        thre_inlier_angle=1 * np.pi / 180,
        thre_inlier_rate=0.9,
        thre_flow_existing_rate=0.1,
        num_ransac=100,
        flowarrow_step_forvis=20,
        flowlength_factor_forvis=1,
        ransac_all_inlier_estimation=False,
        foe_search_step=1,
        log_level=0,
        movprob_lengthfactor_coeff=0.25,
        middle_theta=30 * np.pi / 180,
    ) -> None:
        # constants
        self.LOG_LEVEL = log_level
        self.THRE_FLOWLENGTH = thre_flowlength
        self.THRE_INLIER_ANGLE = thre_inlier_angle
        self.THRE_COS_INLIER = np.cos(self.THRE_INLIER_ANGLE)
        self.THRE_INLIER_RATE = thre_inlier_rate
        self.THRE_FLOW_EXISTING_RATE = thre_flow_existing_rate
        self.NUM_RANSAC = num_ransac
        self.FLOWARROW_STEP_FORVIS = flowarrow_step_forvis
        self.FLOWLENGTH_FACTOR_FORVIS = flowlength_factor_forvis
        self.RANSAC_ALL_INLIER_ESTIMATION = ransac_all_inlier_estimation
        self.SEARCH_STEP = foe_search_step
        self.THRE_FOE_W_INF = 1e-10
        self.MOVPROB_LENGTHFACTOR_COEFF = movprob_lengthfactor_coeff
        self.MIDDLE_THETA = middle_theta

        # variables
        self.state = CameraState.ROTATING  # most unkown movement.
        self.flow_existing_rate_in_static = 0.0
        self.mean_flow_length_in_static = 0.0
        self.inlier_rate = 0.0
        self.foe_hom = None
        self.foe_sign = 1  # +1 for source (the flow is expanding, which means camera is toward front), -1 for sink
        self.foe_camstate_img = None
        self.moving_prob = None  # 0: inlier=stop, 1: outlier=moving
        self.moving_prob_img = None
        self.intermediate_foe_img = None

        self.logger = logging.getLogger(__name__)

    def compute(self, flow, sky_mask, static_mask):
        """..ext.
        compute focus of expansion from optical flow.
        args:
            flow: optical flow. shape = (height, width, 2): 2 channel corresponds to (u, v)
            sky_mask: mask of sky. shape = (height, width), dtype = bool.
            static_mask: mask of static (very low prior moving probability segment) object except sky like grounds. shape = (height, width), dtype = bool
        """
        self.flow = flow
        self.sky_mask = sky_mask
        self.static_mask = static_mask

        self.prepare_variables()

        self.comp_flow_existing_rate_in_static()

        if self.flow_existing_rate_in_static < self.THRE_FLOW_EXISTING_RATE:
            self.state = CameraState.STOPPING
            self.comp_flow_existence()
        else:
            # camera is considered as moving (set as rotating by default)
            self.comp_foe_by_ransac()
            self.comp_movpixprob(self.foe_hom)

    def draw(self, bg_img=None):
        self.bg_img = bg_img

        self.prepare_canvas()
        self.draw_flowarrow(self.flow, self.foe_camstate_img)
        self.draw_state()
        self.draw_moving_prob()

    def prepare_variables(self):
        self.state = CameraState.ROTATING
        self.moving_prob = (
            np.ones((self.flow.shape[0], self.flow.shape[1]), dtype=np.float16) * 0.5
        )
        self.moving_prob[self.sky_mask] = 0.0
        self.foe_hom = None
        self.foe_sign = 1

    def comp_flow_existing_rate_in_static(self):
        num_flow_existing_pix_in_static = 0

        # check pixels inside static mask
        staticpix_indices = np.where(self.static_mask)
        sum_flow_length = 0.0
        for i in range(len(staticpix_indices[0])):
            row = staticpix_indices[0][i]
            col = staticpix_indices[1][i]

            # get flow
            u = self.flow[row, col, 0]
            v = self.flow[row, col, 1]

            # count flow existing pix in static mask, and compute temporary moving prob.
            flow_length = np.sqrt(u**2 + v**2)
            sum_flow_length += flow_length
            if flow_length > self.THRE_FLOWLENGTH:
                num_flow_existing_pix_in_static += 1

        if len(staticpix_indices[0]) == 0:
            self.mean_flow_length_in_static = 0
            self.flow_existing_rate_in_static = 0
        else:
            self.mean_flow_length_in_static = sum_flow_length / len(
                staticpix_indices[0]
            )
            self.flow_existing_rate_in_static = num_flow_existing_pix_in_static / len(
                staticpix_indices[0]
            )

        if self.LOG_LEVEL > 0:
            print(
                f"[INFO] mean flow length in static: {self.mean_flow_length_in_static}"
            )
            print(
                f"[INFO] flow existing pixel rate: {self.flow_existing_rate_in_static * 100:.2f} %"
            )

    def comp_flow_existence(self):
        # at this moment, all flow existing pixel will be set as moving.
        # check pixels inside non static mask
        for row in range(self.flow.shape[0]):
            for col in range(self.flow.shape[1]):
                # get flow
                u = self.flow[row, col, 0]
                v = self.flow[row, col, 1]

                flow_length = np.sqrt(u**2 + v**2)
                # we use the similar probability as camera is moving case.
                self.moving_prob[row, col] = min(
                    1.0, abs(np.log10(1 + abs(flow_length / self.THRE_FLOWLENGTH)))
                )

    def comp_foe_by_ransac(self):
        """
        compute FoE by RANSAC, determine it's sign.
        """
        # if we set below as 0.0, it caused a error in self.foe = self._comp_crosspt().
        self.inlier_rate = 1e-6
        self.foe_hom = None
        self.foe_sign = 1
        best_inlier_mat = None
        if self.LOG_LEVEL > 3:
            # need to reset every time because it might be pressed 'q' to dkip the previous drawing image.
            self.display_foe_flow_arrow_img = True

        for try_num in range(self.NUM_RANSAC):
            foe_candi_hom, foe_candi_sign = self.comp_foe_candidate()
            if foe_candi_hom is None:  # this will unlikely happen.
                print(
                    f"[WARNING] foe_candi is None in ransac {try_num} trial. @ comp_foe_by_ransac"
                )
                continue

            inlier_rate_candi, inlier_mat_candi = self.comp_inlier_rate(
                foe_candi_hom, foe_candi_sign
            )

            if inlier_rate_candi > self.inlier_rate:
                # update by the current best
                self.inlier_rate = inlier_rate_candi
                best_inlier_mat = inlier_mat_candi
                self.foe_hom = foe_candi_hom
                self.foe_sign = foe_candi_sign

                # stop if inlier rate is high enough
                if self.inlier_rate > self.THRE_INLIER_RATE:
                    self.state = CameraState.ONLY_TRANSLATING
                    break

        if self.RANSAC_ALL_INLIER_ESTIMATION and (best_inlier_mat.shape[0] > 1):
            if self.LOG_LEVEL > 2:
                self.intermediate_foe_img.fill(0)  # Clear previous drawings
                skip_step = max(1, best_inlier_mat.shape[0] // 100)
                for line in best_inlier_mat[::skip_step]:
                    self.draw_line(line, self.intermediate_foe_img)
                cv2.imshow("Debug", self.intermediate_foe_img)

            refined_foe_hom = self._comp_crosspt(best_inlier_mat)

            if refined_foe_hom is not None:
                # we need to refine FoE sign too.
                # Because if FoE is very far and flow is almost parallel case, and the candidate and refined FoE
                # is at the opposite direction case, the sign must be changed.
                is_far_foe = False
                if refined_foe_hom[2] != 0 and self.foe_hom[2] != 0:
                    refined_foe = refined_foe_hom[0:2] / refined_foe_hom[2]
                    best_foe_candi = self.foe_hom[0:2] / self.foe_hom[2]
                    refined_dist = np.linalg.norm(refined_foe - best_foe_candi)
                    # Let's use image withth+hight as a threshold.
                    if refined_dist > (self.flow.shape[0] + self.flow.shape[1]):
                        is_far_foe = True
                    if is_far_foe and np.dot(refined_foe, best_foe_candi) < 0:
                        # if the sign is changed, we need to change the sign.
                        self.foe_sign *= -1
                        if self.LOG_LEVEL > 0:
                            print(
                                f"[INFO] best FoE candidate: {best_foe_candi}, refined FoE: {refined_foe}, "
                                f"[INFO] (refinedFoE, bestFoEcandi) = {np.dot(refined_foe, best_foe_candi)}, so FoE signe is flipped."
                            )

                self.foe_hom = refined_foe_hom

            if self.LOG_LEVEL > 0 and (
                abs(refined_foe_hom[2]) >= self.THRE_FOE_W_INF
            ):  # check distance from foe_candi to foe
                print(
                    f"[INFO] RANSAC all inlier estimation: "
                    f"best FoE candidate: {best_foe_candi}, "
                    f"refined FoE: {refined_foe}, "
                    f"refined FoE sign: {self.foe_sign}, "
                    f"distance from best FoE_candi to FoE [pix] = "
                    f"{refined_dist}"
                )

    def _get_random_flowline(
        self, max_retries=1000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pick a random pixel from the static mask until a valid flow line is found.
        Returns:
            line: flow line in 3D homogeneous coordinate.
            flow pt: flow destination point (row, col)
            flow: flow at the pixel.
        """
        for _ in range(max_retries):
            row, col = self.random_point_in_static_mask()
            line = self.comp_flowline(row, col)
            flow_pt = np.array([col, row])
            if line is not None:
                return line, flow_pt, self.flow[row, col]
        raise RuntimeError(
            "[ERROR] Failed to find a valid flow line after {max_retry} attempts @ comp_foe_candidate"
        )

    def comp_foe_candidate(self) -> tuple[np.ndarray, int]:
        """
        compute FoE from 2 flow lines only inside static mask.
        FoE is in 3D homogeneous coordinate.
        If candidate is None, reselect 2 flow lines.
        The camera is considered not stopping when this function is called, so there must exist 2 flow lines.
        return:
            foe_candi_hom: candidate FoE in 3D homogeneous coordinate.
            foe_candi_sign: sign of candidate FoE. positive means source of the optical flow. 0: inifinity.
        """
        # Get two valid flow lines and compute their cross product as the candidate FoE.
        l1, pt1, flow1 = self._get_random_flowline()
        l2, pt2, flow2 = self._get_random_flowline()
        foe_candi_hom = np.cross(l1, l2)

        # draw debug image
        if self.LOG_LEVEL > 2:
            self.intermediate_foe_img = np.zeros(
                (self.flow.shape[0], self.flow.shape[1], 3), dtype=np.uint8
            )
            self.draw_line(l1, self.intermediate_foe_img)
            self.draw_line(l2, self.intermediate_foe_img)
            self.draw_homogeneous_point(foe_candi_hom, self.intermediate_foe_img)
            cv2.imshow("Debug", self.intermediate_foe_img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                exit()

        if abs(foe_candi_hom[2]) < self.THRE_FOE_W_INF:
            # FoE is at inifinity case
            # pt1,pt2 can be considered at the center (0,0).
            pt1 = np.array([0, 0])
            pt2 = np.array([0, 0])
            foe_candi = foe_candi_hom[0:2] * np.sign(foe_candi_hom[2])
        else:
            foe_candi = foe_candi_hom[0:2] / foe_candi_hom[2]

        # compute foe_candi sign
        inner1 = np.inner(flow1, pt1 - foe_candi)
        inner2 = np.inner(flow2, pt2 - foe_candi)
        if inner1 > 0 and inner2 > 0:
            foe_candi_sign = 1
        elif inner1 < 0 and inner2 < 0:
            foe_candi_sign = -1
        else:  # this should not happen.
            foe_candi_sign = 1  # actually, it is unbiguous, but this need to be 1 or -1 when computing inlier late, so I  set it as 1.

        return foe_candi_hom, foe_candi_sign

    def comp_flowline(self, row: int, col: int) -> np.ndarray:
        """
        compute flow line from flow at (row, col), and its directive sign.
        args:
            row: row index of flow
            col: column index of flow
        return:
            line: flow line in 3D homogeneous coordinate. if flow is too small, return None.
        """
        x = [col, row, 1]

        # flow
        u = self.flow[row, col, 0]
        v = self.flow[row, col, 1]

        # if flow is too small, return zero line
        flow_lentgh = np.sqrt(u**2 + v**2)
        if flow_lentgh < self.THRE_FLOWLENGTH:
            return None

        x_prev = [col - u, row - v, 1]

        if self.LOG_LEVEL > 2:
            cv2.arrowedLine(
                self.intermediate_foe_img,
                tuple(map(int, x_prev[0:2])),
                tuple(x[0:2]),
                (0, 0, 255),
                3,
            )

        # no rotation correction version
        line = np.cross(x, x_prev)
        return line

    def random_point_in_static_mask(self):
        # Find the indices of all pixels in the static mask that have a value of 1
        indices = np.where(self.static_mask == 1)

        # Randomly select one of the indices
        index = np.random.choice(len(indices[0]))

        # Get the row and column values corresponding to the selected index
        row = indices[0][index]
        col = indices[1][index]

        return row, col

    def _comp_crosspt(self, inlier_mat: np.ndarray) -> np.ndarray | None:
        """
        Compute the most probable crossing point (Focus of Expansion) from the inlier lines.
        The FoE is the point p such that L @ p = 0 for all lines L in inlier_mat.
        This corresponds to the right singular vector of L associated with the smallest singular value.
        Args:
            inlier_mat: 2D array of inlier lines. shape = (num_lines, 3)
        Returns:
            crossing_point_homogeneous: crossing point in 3D homogeneous coordinate  [x,y,w].
                            If w is very small, FoE is at/near infinity.
                            Returns None if computation fails..
        """
        # Check that there are enough lines to compute the crossing point
        if (inlier_mat is None) or (inlier_mat.ndim != 2) or (inlier_mat.shape[0] < 2):
            print("[WARNING] Not enough lines to compute the crossing point.")
            return None
        if inlier_mat.shape[1] != 3:
            if self.LOG_LEVEL > 0:
                print(
                    f"[WARNING] _comp_crosspt: Inlier matrix columns "
                    f"({inlier_mat.shape[1]}) not equal to 3."
                )
            return None

        try:
            # np.linalg.svd returns U, S, Vh (where Vh is V.T or V*)
            # S contains singular values in descending order.
            # The rows of Vh are the right singular vectors.
            # The last row of Vh corresponds to the smallest singular value.
            U, S, Vt = np.linalg.svd(inlier_mat, full_matrices=False)

            if self.LOG_LEVEL > 1:
                print(f"[DEBUG] _comp_crosspt: Singular values: {S}")

            # The crossing point is the last row of Vt
            crossing_point_homogeneous = Vt[-1, :]

            return crossing_point_homogeneous
        except np.linalg.LinAlgError as e:
            if self.LOG_LEVEL > 0:
                print(f"[ERROR] _comp_crosspt: SVD computation failed: {e}")
            return None
        except Exception as e:
            if self.LOG_LEVEL > 0:
                print(f"[ERROR] _comp_crosspt: Unexpected error during SVD: {e}")
            return None

    def _inlier_decision(
        self, row: int, col: int, foe_hom: np.ndarray, foe_sign: int
    ) -> int:
        """
        Decide the input row, col point is inlier or outlier.
        Args:
            row: row index of flow
            col: column index of flow
            foe_hom: FoE in 3D homogeneous coordinate. (u,v,w)
            foe_sign: sign of FoE. positive means source of the optical flow.
        Return:
            inlier_decision: 1: inlier, -1: outlier, 0: unknown.
        """
        # get flow
        flow_u = self.flow[row, col, 0]
        flow_v = self.flow[row, col, 1]
        flow_length = np.sqrt(flow_u**2 + flow_v**2)

        if flow_length < self.THRE_FLOWLENGTH:
            return 0  # unknown, because the flow is too small.

        if abs(foe_hom[2]) < self.THRE_FOE_W_INF:
            # in case FoE is infinite, point can be considered as 0,0. foe2pt -> (0,0)-foe.
            row = 0
            col = 0
            foe_u = foe_hom[0] * np.sign(foe_hom[2])
            foe_v = foe_hom[1] * np.sign(foe_hom[2])
        else:
            foe_u = foe_hom[0] / foe_hom[2]
            foe_v = foe_hom[1] / foe_hom[2]

        # check the angle between flow and FoE-to-each-pixel is lower than the threshold.
        foe2pt = np.array([col - foe_u, row - foe_v])
        foe2pt_length = np.sqrt(foe2pt[0] ** 2 + foe2pt[1] ** 2)
        foe2pt_length = max(foe2pt_length, 1e-6)  # avoid division by zero
        cos_foe_flow = np.dot(
            foe_sign * foe2pt,
            (flow_u, flow_v),
        ) / (foe2pt_length * flow_length)
        cos_foe_flow = np.clip(cos_foe_flow, -1.0, 1.0)

        # return as an inlier when the angle is close.
        if cos_foe_flow > self.THRE_COS_INLIER:
            return 1
        else:
            return -1

    def comp_inlier_rate(
        self, foe_candi_hom: np.ndarray, foe_candi_sign: int
    ) -> tuple[float, np.ndarray | None]:
        """
        compute inlier rate except sky mask from a candidate FoE.
        args:
            foe_candi_hom: FoE candidate in 3D homogeneous coordinate
            foe_candi_sign: sign of candidate FoE. positive means source of the optical flow.
        The sign of the candidate FoE is used to determine the direction of the flow.
        Returns:
            tuple: (inlier_rate, inlier_flowlines_mat)
                inlier_rate: inlier rate of the candidate FoE.
                inlier_flowlines_mat: inlier flow lines in 3D homogeneous coordinate.
        """
        num_inlier = 0
        num_outlier = 0
        inlier_flowlines_mat = None

        # check pixels inside flow existing static mask area.
        for row in range(0, self.static_mask.shape[0], self.SEARCH_STEP):
            for col in range(0, self.static_mask.shape[1], self.SEARCH_STEP):
                if self.static_mask[row, col]:
                    decision = self._inlier_decision(
                        row, col, foe_candi_hom, foe_candi_sign
                    )  # check if the pixel is inlier or outlier.

                    if decision == 1:
                        num_inlier += 1

                        # add inlier flow line vector to matrix as a row vector to compute FoE by RANSAC
                        x_current = np.array([col, row, 1])
                        x_prev = np.array(
                            [
                                col - self.flow[row, col, 0],
                                row - self.flow[row, col, 1],
                                1,
                            ]
                        )
                        flowline = np.cross(x_current, x_prev)
                        if inlier_flowlines_mat is None:
                            inlier_flowlines_mat = flowline.reshape(1, 3)
                        else:
                            inlier_flowlines_mat = np.vstack(
                                (inlier_flowlines_mat, flowline.reshape(1, 3))
                            )

                        if self.LOG_LEVEL > 3 and self.display_foe_flow_arrow_img:
                            if abs(foe_candi_hom[2]) < self.THRE_FOE_W_INF:
                                # use simpified version because this is just debug image.
                                w = self.THRE_FOE_W_INF * np.sign(foe_candi_hom[2])
                            else:
                                w = foe_candi_hom[2]
                            foe_u = foe_candi_hom[0] / w
                            foe_v = foe_candi_hom[1] / w

                            self._show_foe_flow_arrow_img(
                                row,
                                col,
                                foe_u,
                                foe_v,
                                foe_candi_sign,
                                self.flow[row, col, 0],
                                self.flow[row, col, 1],
                            )
                    elif decision == -1:
                        num_outlier += 1

        # Calculate results
        inlier_rate = 0.0
        num_flow_existingpix = num_inlier + num_outlier
        if num_flow_existingpix > 0:
            inlier_rate = num_inlier / num_flow_existingpix

        if self.LOG_LEVEL > 0:
            foe_uvcoordi = foe_candi_hom[0:2] / foe_candi_hom[2]
            print(
                f"[INFO] FoE candidate = {foe_uvcoordi}, FoE sign = {foe_candi_sign}, inlier rate: {inlier_rate * 100:.2f} %,"
            )

        return inlier_rate, inlier_flowlines_mat

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def comp_movpixprob(self, foe) -> float:
        """
        compute moving pixel probability by length and angle difference.
        args:
            foe: FoE in 3D homogeneous coordinate
        """
        if foe is None:
            raise ValueError("FoE is None @ comp_movpixprob")

        # treat candidate FoE is infinite case
        if abs(foe[2]) < 1e-10:
            foe[2] = 1e-10
        foe_u = foe[0] / foe[2]
        foe_v = foe[1] / foe[2]

        for row in range(0, self.flow.shape[0]):
            for col in range(0, self.flow.shape[1]):
                # get flow
                flow_u = self.flow[row, col, 0]
                flow_v = self.flow[row, col, 1]
                flow_length = np.sqrt(flow_u**2 + flow_v**2)
                # compute length factor for adding length difference when the angle difference is small case.
                if self.mean_flow_length_in_static < self.THRE_FLOWLENGTH:
                    mean_length = self.THRE_FLOWLENGTH
                else:
                    mean_length = self.mean_flow_length_in_static
                length_factor = abs(
                    np.log10(1 + abs(flow_length - mean_length))
                )  # totally the same case should bo 0 = log(1).

                # check the angle between flow and expected flow direction (signed FoE-to-each-pixel) is lower than the threshold.
                expect_flowdir = self.foe_sign * np.array([col - foe_u, row - foe_v, 1])
                expect_flowdir_length = np.sqrt(
                    expect_flowdir[0] ** 2 + expect_flowdir[1] ** 2
                )
                # avoid division by zero
                expect_flowdir_length = max(expect_flowdir_length, self.THRE_FOE_W_INF)
                if flow_length < self.THRE_FLOWLENGTH:
                    cos_foe_flow = 1.0  # let it be the same angle because it cannot compute angle difference.
                else:
                    cos_foe_flow = np.dot(
                        (expect_flowdir[0], expect_flowdir[1]), (flow_u, flow_v)
                    ) / (expect_flowdir_length * flow_length)
                # map cosine difference to moving probability.
                angle_diff_prob = min(
                    np.arccos(cos_foe_flow) / (2 * self.MIDDLE_THETA),
                    1.0,
                )

                # finally add length factor to moving probability
                self.moving_prob[row, col] = np.clip(
                    angle_diff_prob + self.MOVPROB_LENGTHFACTOR_COEFF * length_factor,
                    0.0,
                    1.0,
                )

        # Ensure sky mask remains 0 probability after calculations
        self.moving_prob[self.sky_mask] = 0.0

    def _show_foe_flow_arrow_img(
        self, row, col, foe_u, foe_v, foe_sign, flow_u, flow_v
    ):
        LENGTH_FACTOR = 10
        foe_flow_img = self.intermediate_foe_img.copy()

        # paint self.static_mask area as gray
        foe_flow_img[self.static_mask] = [128, 128, 128]

        # draw arrows.
        if foe_sign > 0:
            cv2.arrowedLine(
                foe_flow_img,
                (int(foe_u), int(foe_v)),
                (col, row),
                (0, 255, 0),
                3,
            )
        else:
            # FoE sign is negative case. the flow direction is point to FoE.
            cv2.arrowedLine(
                foe_flow_img,
                (col, row),
                (int(foe_u), int(foe_v)),
                (0, 255, 0),
                3,
            )
        cv2.arrowedLine(
            foe_flow_img,
            (col, row),
            (int(col + flow_u * LENGTH_FACTOR), int(row + flow_v * LENGTH_FACTOR)),
            (0, 0, 255),
            3,
        )
        cv2.putText(
            foe_flow_img,
            "you can close this image by pressing 'q'",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("FoE and flow", foe_flow_img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            self.display_foe_flow_arrow_img = False
            cv2.destroyWindow("FoE and flow")

    def prepare_canvas(self):
        if self.bg_img is None:
            self.foe_camstate_img = np.zeros(
                (self.flow.shape[0], self.flow.shape[1], 3), dtype=np.uint8
            )
        else:
            self.foe_camstate_img = self.bg_img.copy()
            if self.LOG_LEVEL > 2:
                self.intermediate_foe_img = self.bg_img.copy()

    def draw_flowarrow(self, flow, img):
        """
        draw flow as arrow, colored by green:inlier, red:outlier, gray: others.
        args:
            flow: optical flow. shape = (height, width, 2): 2 channel corresponds to (u, v)
            img: image to draw on.
        """
        for row in range(0, flow.shape[0], self.FLOWARROW_STEP_FORVIS):
            for col in range(0, flow.shape[1], self.FLOWARROW_STEP_FORVIS):
                if self.foe_hom is None:
                    self.logger.warning("[WARNING] FoE is None.")
                    decision = 0
                else:
                    decision = self._inlier_decision(
                        row, col, self.foe_hom, self.foe_sign
                    )

                if decision == 1:
                    color = (0, 255, 0)  # inlier: green
                elif decision == -1:
                    color = (0, 0, 255)  # outlier: red
                else:
                    color = (128, 128, 128)  # unknown: gray

                u = flow[row, col, 0]
                v = flow[row, col, 1]
                cv2.arrowedLine(
                    img,
                    pt1=(
                        int(col - u * self.FLOWLENGTH_FACTOR_FORVIS),
                        int(row - v * self.FLOWLENGTH_FACTOR_FORVIS),
                    ),
                    pt2=(col, row),
                    color=color,
                    thickness=3,
                )

    def draw_homogeneous_point(self, hom_pt, out_img):
        if hom_pt[2] == 0:
            return
        pt = (int(hom_pt[0] / hom_pt[2]), int(hom_pt[1] / hom_pt[2]))
        # draw cross at the point
        cv2.line(out_img, (pt[0] - 10, pt[1]), (pt[0] + 10, pt[1]), (0, 0, 255), 10)
        cv2.line(out_img, (pt[0], pt[1] - 10), (pt[0], pt[1] + 10), (0, 0, 255), 10)

    def draw_line(self, line, img):
        if line[0] == 0 and line[1] == 0:
            return
        pt1 = (0, int(-line[2] / line[1]))
        pt2 = (img.shape[1], int(-(line[2] + line[0] * img.shape[1]) / line[1]))
        cv2.line(img, pt1, pt2, (0, 255, 0), 1)

    def draw_state(self):
        if self.state == CameraState.STOPPING:
            cv2.putText(
                self.foe_camstate_img,
                "Camera is stopping",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
        elif self.state == CameraState.ONLY_TRANSLATING:
            self.draw_homogeneous_point(self.foe_hom, self.foe_camstate_img)
            cv2.putText(
                self.foe_camstate_img,
                "Camera is only translating",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
        elif self.state == CameraState.ROTATING:
            cv2.putText(
                self.foe_camstate_img,
                "Camera is rotating",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

    def draw_moving_prob(self):
        if self.moving_prob is None:
            return
        self.moving_prob_img = cv2.applyColorMap(
            np.uint8(self.moving_prob * 255), cv2.COLORMAP_JET
        )
        cv2.putText(
            self.moving_prob_img,
            "FoE based likelihood",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
