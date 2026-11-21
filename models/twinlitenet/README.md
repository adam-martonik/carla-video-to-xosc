# TwinLiteNet (lane lines + drivable area)

Used by `agents/lane_lines.py` to segment lane lines and the drivable area of dashcam frames (640x360 RGB input, ~40 ms per
frame on CPU). Without these files the lane detector falls back to a classical OpenCV filter.

- Source: https://github.com/chequanghuy/TwinLiteNet (MIT licence), trained on BDD100K
- `best.pth`: https://github.com/chequanghuy/TwinLiteNet/raw/refs/heads/main/pretrained/best.pth
  sha256 `98ffc6c8abc98fda72052117249ce4e2dbc85e58cfcfc6ee2d42f5398c772463`
- `TwinLite.py`: https://raw.githubusercontent.com/chequanghuy/TwinLiteNet/main/model/TwinLite.py (unchanged)
  sha256 `dd0f4e7f390f6468d692596543a60cdb219eb26924fb956df8a91ff716268c27`

The weights are loaded with `torch.load(..., weights_only=True)`, so the file can only hold tensors.
