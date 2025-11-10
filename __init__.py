from .dfloat11_model_loader import DFloat11ModelLoader, DFloat11DiffusersModelLoader, DFloat11ModelCompressor, DFloat11CheckpointCompressor, CheckpointLoaderWithDFloat11

NODE_CLASS_MAPPINGS = {
    "DFloat11ModelLoader": DFloat11ModelLoader,
    "DFloat11DiffusersModelLoader": DFloat11DiffusersModelLoader,
    "DFloat11ModelCompressor": DFloat11ModelCompressor,
    "DFloat11CheckpointCompressor": DFloat11CheckpointCompressor,
    "CheckpointLoaderWithDFloat11": CheckpointLoaderWithDFloat11,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DFloat11ModelLoader": "DFloat11 Model Loader",
    "DFloat11DiffusersModelLoader": "DFloat11 diffusers-native Model Loader",
    "DFloat11ModelCompressor": "DFloat11 Model Compressor",
    "DFloat11CheckpointCompressor": "DFloat11 Checkpoint Compressor",
    "CheckpointLoaderWithDFloat11": "Load Checkpoint with DFloat11 Unet",
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
