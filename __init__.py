from .dfloat11_model_loader import DFloat11ModelLoader, DFloat11ModelLoaderAdvanced, DFloat11DiffusersModelLoader, DFloat11ModelCompressor, DFloat11CheckpointCompressor, CheckpointLoaderWithDFloat11, DFloat11LoadingPatch, DFloat11Decompressor

NODE_CLASS_MAPPINGS = {
    "DFloat11ModelLoader": DFloat11ModelLoader,
    "DFloat11ModelLoaderAdvanced": DFloat11ModelLoaderAdvanced,
    "DFloat11DiffusersModelLoader": DFloat11DiffusersModelLoader,
    "DFloat11ModelCompressor": DFloat11ModelCompressor,
    "DFloat11CheckpointCompressor": DFloat11CheckpointCompressor,
    "CheckpointLoaderWithDFloat11": CheckpointLoaderWithDFloat11,
    "DFloat11LoadingPatch": DFloat11LoadingPatch,
    "DFloat11Decompressor": DFloat11Decompressor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DFloat11ModelLoader": "DFloat11 Model Loader",
    "DFloat11ModelLoaderAdvanced": "DFloat11 Model Loader (Advanced)",
    "DFloat11DiffusersModelLoader": "DFloat11 diffusers-native Model Loader",
    "DFloat11ModelCompressor": "DFloat11 Model Compressor",
    "DFloat11CheckpointCompressor": "DFloat11 Checkpoint Compressor",
    "CheckpointLoaderWithDFloat11": "Load Checkpoint with DFloat11 Unet",
    "DFloat11LoadingPatch" : "Patch DFloat11 Loading",
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
