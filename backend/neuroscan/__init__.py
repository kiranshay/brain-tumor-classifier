"""NeuroScan: modular brain-tumor classifier package.

Submodules:
    config           - YAML config loader
    logging_setup    - Python logging configuration
    data             - MRI preprocessing, transforms, dataloaders
    models           - EfficientNet builder + checkpoint loader (with assertions)
    gradcam          - Grad-CAM heatmap generation
    inference        - Classifier abstraction and cascade inference
    evaluation       - Per-class metrics, validation loop
    training         - Training loop driven by config
"""
