import os

DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def resolve_device(env_name: str, auto_mps: bool = True) -> str:
    import torch

    choice = os.environ.get(env_name, "auto").strip().lower() or "auto"
    if choice not in DEVICE_CHOICES:
        print(f"[device] {env_name}={choice!r} is not one of {DEVICE_CHOICES}; using auto")
        choice = "auto"
    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()  # Apple Silicon GPU
    if choice == "cuda" and not cuda:
        print(f"[device] {env_name}=cuda but no CUDA GPU is available; using cpu")
        return "cpu"
    if choice == "mps" and not mps:
        print(f"[device] {env_name}=mps but no Apple GPU is available; using cpu")
        return "cpu"
    if choice == "auto":
        return "cuda" if cuda else "mps" if mps and auto_mps else "cpu"
    return choice
