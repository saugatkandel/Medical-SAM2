from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from argparse import ArgumentParser  # noqa: I001
    import numpy as np

torch.backends.cudnn.benchmark = True


class MedicalSam2ImagePredictor:
    def __init__(self, net: nn.Module, args: ArgumentParser):
        self.net = net
        self.args = args

        self.gpu_device = torch.device("cuda", args.gpu_device)
        self.pos_weight = torch.ones([1]).cuda(device=self.gpu_device) * 2

        # use bfloat16 for the entire notebook
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.feat_sizes = [(256, 256), (128, 128), (64, 64)]

        self.threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
        self.memory_bank_list = []

        self._is_image_set = False
        self._features = None
        # Whether the predictor is set for single image or a batch of images
        self._is_batch = False
        self._batch_size = None

        self.net.eval()

    @torch.no_grad()
    def set_image_batch(self, image_list: Sequence[np.ndarray | torch.Tensor]):
        self.reset_predictor()
        images = torch.as_tensor(image_list, dtype=torch.float32, device=self.gpu_device)
        backbone_out = self.net.forward_image(images)
        _, vision_feats, vision_pos_embeds, _ = self.net._prepare_backbone_features(backbone_out)
        self._batch_size = vision_feats[-1].size(1)

        to_cat_memory = []
        to_cat_memory_pos = []
        to_cat_image_embed = []

        """ memory condition """
        if len(self.memory_bank_list) == 0:
            vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(
                torch.zeros(1, self._batch_size, self.net.hidden_dim)
            ).to(device="cuda")
            vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(
                torch.zeros(1, self._batch_size, self.net.hidden_dim)
            ).to(device="cuda")

        else:
            for element in self.memory_bank_list:
                maskmem_features = element[0]
                maskmem_pos_enc = element[1]
                to_cat_memory.append(
                    maskmem_features.cuda(non_blocking=True).flatten(2).permute(2, 0, 1)
                )
                to_cat_memory_pos.append(
                    maskmem_pos_enc.cuda(non_blocking=True).flatten(2).permute(2, 0, 1)
                )
                to_cat_image_embed.append((element[3]).cuda(non_blocking=True))  # image_embed

            memory_stack_ori = torch.stack(to_cat_memory, dim=0)
            memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
            image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)

            vision_feats_temp = vision_feats[-1].permute(1, 0, 2).view(self._batch_size, -1, 64, 64)
            # vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, 64, 64)
            vision_feats_temp = vision_feats_temp.reshape(self._batch_size, -1)

            image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
            vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
            similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()

            similarity_scores = F.softmax(similarity_scores, dim=1)
            sampled_indices = torch.multinomial(
                similarity_scores, num_samples=self._batch_size, replacement=True
            ).squeeze(1)  # Shape [batch_size, 16]

            memory_stack_ori_new = memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3)
            memory = memory_stack_ori_new.reshape(
                -1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3)
            )

            memory_pos_stack_new = (
                memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3)
            )
            memory_pos = memory_pos_stack_new.reshape(
                -1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3)
            )

            vision_feats[-1] = self.net.memory_attention(
                curr=[vision_feats[-1]],
                curr_pos=[vision_pos_embeds[-1]],
                memory=memory,
                memory_pos=memory_pos,
                num_obj_ptr_tokens=0,
            )

        feats = [
            feat.permute(1, 2, 0).view(self._batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self.feat_sizes[::-1])
        ][::-1]
        image_embed = feats[-1]
        high_res_feats = feats[:-1]

        self._is_image_set = True
        self._is_batch = True
        self._features = (vision_feats, image_embed, high_res_feats)

    @torch.no_grad()
    def predict_batch(
        self,
        point_coords_batch: Sequence[np.ndarray] | None = None,
        point_labels_batch: Sequence[np.ndarray] | None = None,
        mask_threshold: float = 0.5,
    ):
        """Currently not supporting boxes and masks."""
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image_batch(...) before mask prediction."
            )
        if (point_coords_batch is None) ^ (point_labels_batch is None):
            # This is the xor operation
            raise ValueError(
                "Both point_coords_batch and point_labels_batch must be provided together."
            )
        elif (point_coords_batch is not None) and (point_labels_batch is not None):
            coords_torch = torch.as_tensor(
                point_coords_batch, dtype=torch.float, device=self.gpu_device
            )
            labels_torch = torch.as_tensor(
                point_labels_batch, dtype=torch.int, device=self.gpu_device
            )
            points = (coords_torch, labels_torch)
            is_mask_from_points = True
        else:
            points = None
            is_mask_from_points = False

        """ prompt encoder """

        sparse_embeddings, dense_embeddings = self.net.sam_prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
            batch_size=self._batch_size,
        )
        vision_feats, image_embed, high_res_feats = self._features
        low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = (
            self.net.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=self.net.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_feats,
            )
        )

        # prediction
        pred = F.interpolate(low_res_multimasks, size=(self.args.out_size, self.args.out_size))
        pred_mask = pred > mask_threshold
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.args.image_size, self.args.image_size),
            mode="bilinear",
            align_corners=False,
        )

        """ memory encoder """
        maskmem_features, maskmem_pos_enc = self.net._encode_new_memory(
            current_vision_feats=vision_feats,
            feat_sizes=self.feat_sizes,
            pred_masks_high_res=high_res_multimasks,
            is_mask_from_pts=is_mask_from_points,
        )

        maskmem_features = maskmem_features.to(torch.bfloat16)
        maskmem_features = maskmem_features.to(device=self.gpu_device, non_blocking=True)
        maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16)
        maskmem_pos_enc = maskmem_pos_enc.to(device=self.gpu_device, non_blocking=True)

        """ memory bank """
        if len(self.memory_bank_list) < 16:
            for batch in range(maskmem_features.size(0)):
                self.memory_bank_list.append([
                    (maskmem_features[batch].unsqueeze(0)),
                    (maskmem_pos_enc[batch].unsqueeze(0)),
                    iou_predictions[batch, 0],
                    image_embed[batch].reshape(-1).detach(),
                ])

        else:
            for batch in range(maskmem_features.size(0)):
                memory_bank_maskmem_features_flatten = [
                    element[0].reshape(-1) for element in self.memory_bank_list
                ]
                memory_bank_maskmem_features_flatten = torch.stack(
                    memory_bank_maskmem_features_flatten
                )

                memory_bank_maskmem_features_norm = F.normalize(
                    memory_bank_maskmem_features_flatten, p=2, dim=1
                )
                current_similarity_matrix = torch.mm(
                    memory_bank_maskmem_features_norm, memory_bank_maskmem_features_norm.t()
                )

                current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                current_similarity_matrix_no_diag[diag_indices, diag_indices] = float("-inf")

                single_key_norm = F.normalize(
                    maskmem_features[batch].reshape(-1), p=2, dim=0
                ).unsqueeze(1)
                similarity_scores = torch.mm(
                    memory_bank_maskmem_features_norm, single_key_norm
                ).squeeze()
                min_similarity_index = torch.argmin(similarity_scores)
                max_similarity_index = torch.argmax(
                    current_similarity_matrix_no_diag[min_similarity_index]
                )

                if (
                    similarity_scores[min_similarity_index]
                    < current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]
                ):
                    if (
                        iou_predictions[batch, 0]
                        > self.memory_bank_list[max_similarity_index][2] - 0.1
                    ):
                        self.memory_bank_list.pop(max_similarity_index)
                        self.memory_bank_list.append([
                            (maskmem_features[batch].unsqueeze(0)),
                            (maskmem_pos_enc[batch].unsqueeze(0)),
                            iou_predictions[batch, 0],
                            image_embed[batch].reshape(-1).detach(),
                        ])
        return pred_mask, pred, high_res_multimasks

    def reset_predictor(self):
        self._is_image_set = False
        self._features = None
        self._is_batch = False
        self._batch_size = None
