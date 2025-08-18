import torch
import torch.nn as nn

class BaseModel(nn.Module):

    def __init__(self, device=None):
        super().__init__()
        self.device = device


    def summary(self, input_shape):
        #input_shape = (1,1,2001)
        #from torchsummaryX import summary
        from torchinfo import summary
        print("\n================================================================")
        print("SUMMARY FOR MODEL:", type(self).__name__)
        print("Input dimensions:", input_shape)
        print("================================================================")
        print(str(self.device))
        device = 'cuda' if 'cuda' in str(self.device) else 'cpu'
        #import code
        #code.interact(local=locals())
        summary(model=self, input_size=input_shape, device=device)
        print(flush=True)

    def additional_outputs(self):
        """For additional logging during evaluation, return a dict of numpy arrays"""
        return None

    @staticmethod
    def _extract_state_dict(ckpt):
        if isinstance(ckpt, dict):
            for k in ("state_dict", "model_state_dict", "model", "net"):
                if isinstance(ckpt.get(k), dict):
                    return ckpt[k]
            return ckpt
        return ckpt

    def load_checkpoint(self, chkp_path, strict=False, map_location="cpu"):
        ckpt = torch.load(chkp_path, map_location=map_location)
        sd = self._extract_state_dict(ckpt)
        if any(k.startswith("module.") for k in sd):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        missing, unexpected = self.load_state_dict(sd, strict=strict)
        self.to(self.device).eval()
        return missing, unexpected
