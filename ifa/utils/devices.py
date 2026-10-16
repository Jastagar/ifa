import os

DEVICE_CHOICES = ("auto", "cuda", "cpu")


def resolve_device(env_name: str) -> str:
    import torch

    choice = os.environ.get(env_name, "auto").strip().lower() or "auto"
    if choice not in DEVICE_CHOICES:
        print(f"[device] {env_name}={choice!r} is not one of {DEVICE_CHOICES}; using auto")
        choice = "auto"
    cuda = torch.cuda.is_available()
    if choice == "cuda" and not cuda:
        print(f"[device] {env_name}=cuda but no CUDA GPU is available; using cpu")
        return "cpu"
    if choice == "auto":
        return "cuda" if cuda else "cpu"
    return choice
