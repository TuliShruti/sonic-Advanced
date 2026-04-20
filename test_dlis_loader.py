from sonic_dashboard.loaders.dlis_loader import load_dlis

file_path = r"C:\Users\shrut\Downloads\G1S3_AIT_DSI_NUC_MAINLOG.DLIS"

with open(file_path, "rb") as f:
    data = load_dlis(f)

frame = data["frame_names"][0]
fdata = data["frames"][frame]

print("Frame:", frame)
print("Available waveforms:", list(fdata["waveforms"].keys()))
print("Available semblance:", list(fdata["semblance"].keys()))

print("\nDepth:   ", None if fdata["depth"] is None else fdata["depth"].shape)
print("PWF2:    ", None if fdata["waveforms"].get("PWF2") is None else fdata["waveforms"]["PWF2"].shape)
print("PWF4:    ", None if fdata["waveforms"].get("PWF4") is None else fdata["waveforms"]["PWF4"].shape)
print("SPR4:    ", None if fdata["semblance"].get("SPR4") is None else fdata["semblance"]["SPR4"].shape)
print("SPR2:    ", None if fdata["semblance"].get("SPR2") is None else fdata["semblance"]["SPR2"].shape)
print("Slowness:", None if fdata["slowness"] is None else fdata["slowness"].shape)