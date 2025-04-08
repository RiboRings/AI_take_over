# Aquamonitor project for 2025 Deep Learning course

This repository hosts our work on the Aquamonitor project for the classification of Benthic macroinvertebrates from Finnish lakes.

We used several deep learning architectures to address the task, including:

- [ResNet18 with new weights](https://github.com/RiboRings/AI_take_over/blob/main/models/aquabasic.py)
- Fine-tuned ResNet18 ([Final version](https://github.com/RiboRings/AI_take_over/blob/main/models/aquaresnet.ipynb), [All versions](https://www.kaggle.com/code/giuliobenedetti/aquaresnet))
- Fine-tuned Swin ([Final version](https://github.com/RiboRings/AI_take_over/blob/main/models/aquaswin.ipynb), [All versions](https://www.kaggle.com/code/giuliobenedetti/aquaswin))
- Fine-tuned Inception ([Final version](https://github.com/RiboRings/AI_take_over/blob/main/models/aquainception.ipynb), [All versions](https://www.kaggle.com/code/giuliobenedetti/aquainception))
- Fine-tuned ViT ([Final version](https://github.com/RiboRings/AI_take_over/blob/main/models/aquavit.ipynb), [All versions](https://www.kaggle.com/code/giuliobenedetti/aquamonitor))
- [ReSwine Ensemble model](https://github.com/RiboRings/AI_take_over/blob/main/models/aquaensemble.ipynb)

Our final model (ReSwine ensemble architecture) is provided in the main directory as a Python script: [model.py](https://github.com/RiboRings/AI_take_over/blob/main/model.py)
