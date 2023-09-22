from pathlib import Path
from typing import Optional

import torch
from torchvision import transforms

from .utils import architectures
from .utils.python_patch_extractor.PatchExtractor import PatchExtractor


class Detector(torch.nn.Module):
    def __init__(self):
        super(Detector, self).__init__()

        self._n_patch = 200
        network_class = getattr(architectures, 'EfficientNetB4')
        self._models = [network_class(n_classes=2, pretrained=False) for _ in range(5)]

        self._transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self._cropper = transforms.RandomCrop(128, 128)

    def load_pretrained(self, weights_path: Path) -> None:
        weights_path_list = [weights_path.joinpath(f'method_{x}.pth') for x in 'ABCDE']
        for idx, model in enumerate(self._models):
            state_dict = torch.load(weights_path_list[idx], map_location='cpu')
            model.load_state_dict(state_dict['net'], strict=True)

    def configure(self, device: Optional[str], training: Optional[bool] = None, **kwargs) -> None:
        if device is not None:
            self.to(device)
            for model in self._models:
                model.to(device)

        if training is None:
            return

        if training:
            self.train()
        else:
            self.eval()

        for model in self._models:
            if training:
                model.train()
            else:
                model.eval()

    def forward(self, img_batch: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        device = img_batch.get_device()
        scores = torch.zeros(img_batch.shape[0], device=device)
        transformed = self._transform(img_batch)

        for idx, batched in enumerate(transformed):
            models_scores = torch.zeros(len(self._models), device=device)

            # Patces used by the first model
            patches_zero = [self._cropper(batched) for _ in range(self._n_patch)]
            patches_zero = torch.stack(patches_zero, dim=0)

            # Patches used by the other models, Extractor requires using numpy arrays
            stride_0 = ((((batched.shape[1] - 128) // 20) + 7) // 8) * 8
            stride_1 = (((batched.shape[2] - 128) // 10 + 7) // 8) * 8
            pe = PatchExtractor(dim=(128, 128, 3), stride=(stride_0, stride_1, 3))
            batched = batched.permute(1, 2, 0).cpu().numpy()
            patches = pe.extract(batched)
            patch_list = list(patches.reshape((patches.shape[0] * patches.shape[1], 128, 128, 3)))
            patch_list = [torch.from_numpy(patch).to(device) for patch in patch_list]
            patches_other = torch.stack(patch_list, dim=0).permute(0, 3, 1, 2)

            for model_idx, model in enumerate(self._models):
                if model_idx == 0:
                    patch_scores = model(patches_zero)
                else:
                    patch_scores = model(patches_other)

                # If even a single patch is detected as synthesized, the model considers the image as synthesized
                # and the highest synthesized patch score is used.
                # Otherwise, the negative of the highest real patch score is used.
                patch_predictions = torch.argmax(patch_scores, dim=1)
                detection_idx = torch.any(patch_predictions).to(torch.int)
                scores_maj_voting = patch_scores[:, detection_idx]
                max_score = torch.max(scores_maj_voting)
                models_scores[model_idx] = max_score if detection_idx == 1 else -max_score

            scores[idx] = torch.mean(models_scores)

        sig = scores.sigmoid()
        label = torch.round(sig).to(torch.int)
        return label, sig
