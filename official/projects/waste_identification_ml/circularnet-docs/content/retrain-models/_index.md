# Retrain CircularNet models

CircularNet's models are initially trained with images captured from Material
Recovery Facilities (MRFs). As a result, these open-source models are
specialized in a limited set of materials, which might not fully align with the
specific materials you collect.

To address your specific use case, you can
retrain the models by utilizing a pipeline built with
[Vertex AI](https://cloud.google.com/vertex-ai/docs). This pipeline is ideal for
analyzing materials from diverse sources, adapting to different business
scenarios, or retraining the model with your own images.