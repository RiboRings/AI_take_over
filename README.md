# Aquamonitor project for 2025 Deep Learning course

This repository hosts our work on the Aquamonitor project for the classification of Benthic macroinvertebrates from Finnish lakes.

We used several deep learning architectures to address the task, including:

- [ResNet18 with new weights](https://github.com/RiboRings/AI_take_over/blob/main/aquabasic.py)
- Fine-tuned ResNet18 ([Final version](https://github.com/RiboRings/AI_take_over/blob/main/aquaresnet.ipynb), [All versions](https://www.kaggle.com/code/giuliobenedetti/aquaresnet))
- Fine-tuned Swin ([Final versions](https://github.com/RiboRings/AI_take_over/blob/main/aquaswin.ipynb), [All versions](https://www.kaggle.com/code/giuliobenedetti/aquaswin))
- Fine-tuned Inception ([Final versions](https://github.com/RiboRings/AI_take_over/blob/main/aquainception.ipynb), [All versions](https://www.kaggle.com/code/giuliobenedetti/aquainception))
- Fine-tuned ViT ([Final versions](https://github.com/RiboRings/AI_take_over/blob/main/aquavit.ipynb), [All versions](https://www.kaggle.com/code/giuliobenedetti/aquamonitor))
- [ReSwine Ensemble model](https://github.com/RiboRings/AI_take_over/blob/main/aquaensemble.ipynb)

Our final model (ReSwine ensemble architecture) is also provided as a Python script: [model.py](https://github.com/RiboRings/AI_take_over/blob/main/model.py)
