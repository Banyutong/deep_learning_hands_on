# 交大密院Deep Learning学习手册
# UM-SJTU-JI Deep learning Hands-on Tutorial 
# Session 10: Deep Dive into Real-World Applications

Congratulations on making it to this point! With the knowledge and skills you've acquired through the previous sessions, you have enough competence to undertake a real-world project.

Below are several projects, each accompanied by a detailed description and guidelines to kickstart your journey. Given your proficiency, **select one project** that piques your interest or aligns with your aspirations. Remember that each project provides a unique set of challenges and learning opportunities, so choose one that you find intriguing and dive deep into it!

### Project 1: Facial Recognition with Attributes
- **Dataset:** CelebA ([Dataset Link](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html))
- **Task:** Multi-label classification for facial attributes.
- **Detailed Task Description:**
    - Develop a deep learning model that predicts the presence of 40 binary attributes (like "Wearing Glasses", "Smiling", etc.) from facial images.
    - The objective is to recognize multiple attributes from a single image, making it a multi-label classification problem.
    - Evaluate the model using relevant metrics like F1-Score, considering it’s a multi-label task.
- **Guidance:**
    - Data Exploration: Visualize the images, explore the distribution of different attributes, and analyze class imbalance.
    - Model Architecture: Utilize CNNs, considering their efficacy in image tasks. Employ architectures like ResNets or VGG, potentially pre-trained on ImageNet, as a starting point.
    - Data Augmentation: Employ techniques like rotation, flipping, and scaling to augment the dataset and enhance the model’s robustness.
    - Loss Function: Employ a binary cross-entropy loss or focal loss, taking into consideration the potential class imbalance across attributes.

### Project 2: Speech to Text Conversion
- **Dataset:** Common Voice ([Dataset Link](https://commonvoice.mozilla.org/en/datasets))
- **Task:** Speech recognition to convert spoken words into text.
- **Detailed Task Description:**
    - Construct a model that takes audio files as input and outputs the corresponding textual transcription.
    - The transcription should be as accurate as possible to the spoken content in the audio file.
    - Evaluate the model considering metrics like Word Error Rate (WER) or Character Error Rate (CER).
- **Guidance:**
    - Data Exploration: Visualize spectrograms and listen to a subset of audio files to understand variations in speech, accents, and background noise.
    - Model Architecture: Employ RNNs or Transformers. Considering the sequential nature of audio data, models like DeepSpeech or Wav2Vec could be explored.
    - Preprocessing: Convert raw audio into a more model-friendly format, such as Mel-frequency cepstral coefficients (MFCCs) or spectrograms.
    - Handling Variable Length: Implement mechanisms to deal with variable-length audio and transcription sequences, potentially using padding or bucketing.

### Project 3: Credit Card Fraud Detection
- **Dataset:** Credit Card Fraud ([Dataset Link](https://www.kaggle.com/mlg-ulb/creditcardfraud))
- **Task:** Identify fraudulent transactions based on transaction data.
- **Detailed Task Description:**
    - Develop a model that categorizes transactions into legitimate or fraudulent.
    - Given the sensitive nature of fraud detection, strive for a model that minimizes false negatives without excessively increasing false positives.
    - Evaluate the model using metrics like precision, recall, F1-score, and potentially, Area Under the Precision-Recall Curve (AUC-PR), given the expected class imbalance.
- **Guidance:**
    - Data Exploration: Analyze the distributions and relationships of different features and explore correlations between them. Examine the class imbalance in the dataset.
    - Model Architecture: Employ traditional ML models (like Random Forests or SVMs) or neural networks, analyzing trade-offs between interpretability and predictive power.
    - Class Imbalance: Address this using under-sampling, over-sampling, or employing class weights in the loss function.
    - Feature Engineering: Given the dataset's anonymized features, experimentation with feature interactions or dimensionality reduction might prove insightful.
### Project 4: Gesture Recognition from Videos
- **Dataset:** 20BN-Jester ([Dataset Link](https://20bn.com/datasets/jester))
- **Task:** Identify and classify hand gestures from video sequences.
- **Detailed Task Description:**
    - Build a model that interprets sequences of video frames to identify and classify different hand gestures (e.g., swiping left, swiping right, stopping).
    - Your model should robustly handle various lighting conditions, backgrounds, and hand orientations.
    - Evaluate model performance using classification metrics like accuracy, precision, recall, and F1-score.
- **Guidance:**
    - Data Exploration: Examine the videos, looking at the variety in gestures, lighting, and backgrounds. Understand the class distribution in the dataset.
    - Model Architecture: Explore 3D CNNs or combinations of 2D CNNs and RNNs/LSTMs to capture spatial and temporal dependencies in the video frames.
    - Data Augmentation: Use techniques like cropping, rotation, and temporal slicing to diversify your training data and make the model more robust.
    - Temporal Dynamics: Ensure your model adequately considers the temporal dynamics of gestures, potentially using recurrent layers or temporal convolution layers.

### Project 5: COVID-19 Detection from CT Scans
- **Dataset:** COVID-19 CT Scans ([Dataset Link](https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset))
- **Task:** Distinguish between CT scans with COVID-19 and without.
- **Detailed Task Description:**
    - Construct a model to categorize CT scans as being indicative or not indicative of a COVID-19 infection.
    - Consider challenges related to 3D data and potential class imbalances.
    - Evaluation should focus on classification metrics, with particular attention to recall to minimize false negatives.
- **Guidance:**
    - Data Exploration: Examine CT images, looking for discernible patterns and differences between classes. Ensure understanding of the 3D nature of the data.
    - Model Architecture: Explore 3D CNN architectures, like 3D U-Net, given the volumetric nature of CT scans.
    - Data Preprocessing: Consider normalizing or standardizing the CT scan pixel values and handling differing scan sizes.
    - Clinical Relevance: Ensure that model interpretations and evaluations consider the clinical implications and relevance of predictions.

### Project 6: Age and Gender Prediction
- **Dataset:** UTKFace ([Dataset Link](https://susanqq.github.io/UTKFace/))
- **Task:** Predict age and gender from facial images.
- **Detailed Task Description:**
    - Develop a model that predicts both the age (as a regression task) and gender (as a binary classification task) from facial images.
    - Ensure the model is robust to various lighting, orientations, and facial expressions.
    - Utilize metrics like Mean Absolute Error (MAE) for age prediction and accuracy/precision/recall for gender prediction.
- **Guidance:**
    - Data Exploration: Visualize facial images, understand age distributions, and inspect the balance between gender classes.
    - Model Architecture: Employ CNN architectures, considering their effectiveness in image-related tasks. Investigate models like ResNets or EfficientNets.
    - Multi-Task Learning: Design your network to have shared layers that learn common features and task-specific layers/heads for age and gender prediction.
    - Data Augmentation: Implement techniques like rotation, scaling, and flipping to enhance model generalization and robustness to varied image inputs.

### Project 7: Human Pose Estimation
- **Dataset:** MPII Human Pose ([Dataset Link](http://human-pose.mpi-inf.mpg.de/))
- **Task:** Develop a model that identifies and locates joints/keypoints on human bodies in images.
- **Detailed Task Description:**
    - Create a model that recognizes and pinpoints human joints/keypoints (e.g., ankles, knees, elbows) in images.
    - Ensure robustness to variations in clothing, body size, and partial occlusions.
    - Evaluate using metrics like Percentage of Correct Keypoints (PCK) or Object Keypoint Similarity (OKS).
- **Guidance:**
    - Data Exploration: Investigate the variability in poses, clothing, and image backgrounds. Ensure a keen understanding of annotation formats.
    - Model Architecture: Consider architectures like OpenPose or HRNet which have shown notable performance in human pose estimation tasks.
    - Data Augmentation: Leverage techniques like rotation, scaling, and flipping to generalize across various poses and perspectives.
    - Post-Processing: Implement and tweak post-processing steps like Non-Maximum Suppression (NMS) to refine the keypoints detected by your model.

### Project 8: Reinforcement Learning in Gaming
- **Dataset:** Gym Retro ([Dataset Link](https://github.com/openai/retro))
- **Task:** Train an agent to play a classic video game by maximizing the score/reward.
- **Detailed Task Description:**
    - Develop an agent that learns to play a classic video game, making decisions that maximize cumulative reward.
    - Ensure stability in learning and adaptability to varied game scenarios.
    - Use cumulative reward and episode length as key evaluation metrics.
- **Guidance:**
    - Getting Started: Familiarize yourself with the Gym Retro environment, understanding state representations and available actions.
    - Algorithm Choice: Explore algorithms like Proximal Policy Optimization (PPO) or Deep Q-Networks (DQN) which have demonstrated stability and efficacy in similar contexts.
    - Reward Design: Ensure the reward structure encourages long-term strategy in addition to immediate gains.
    - Exploration-Exploitation: Manage the balance between exploration and exploitation, potentially by adjusting epsilon in epsilon-greedy strategies or employing an entropy bonus.

### Project 9: Human Activity Recognition from Sensor Data
- **Dataset:** UCI HAR ([Dataset Link](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones))
- **Task:** Construct a model to identify human activities (e.g., walking, sitting) based on smartphone sensor data.
- **Detailed Task Description:**
    - Develop a model that interprets sequences of sensor data to classify human activities.
    - Ensure model stability and robustness across different users and environments.
    - Employ classification metrics like accuracy, precision, and recall for evaluation.
- **Guidance:**
    - Data Exploration: Understand the characteristics of different activities in the sensor readings and examine the balance between activity classes.
    - Model Architecture: Consider time-series models like LSTMs or 1D CNNs, given the sequential nature of sensor data.
    - Data Normalization: Ensure that sensor readings are normalized or standardized to prevent scale discrepancies from affecting learning.
    - Feature Engineering: Explore the creation of additional features, like statistical features, that may provide additional insight into activity types.

### Project 10: Music Genre Classification
- **Dataset:** GTZAN Genre Collection ([Dataset Link](http://marsyas.info/downloads/datasets.html))
- **Task:** Classify music clips into genres.
- **Detailed Task Description:**
    - Develop a model that categorizes music clips into predefined genres.
    - Ensure the model is robust to variations in instrumentals, vocals, and tempo within each genre.
    - Evaluate using classification metrics like accuracy, precision, recall, and F1-score.
- **Guidance:**
    - Data Exploration: Listen to sample tracks, understand genre characteristics, and analyze the distribution of genres.
    - Model Architecture: Consider using CNNs on spectrogram images or RNNs/LSTMs to model temporal dependencies in raw audio.
    - Data Preprocessing: Explore converting audio to spectrograms or Mel-frequency cepstral coefficients (MFCCs) for use as model input.
    - Data Augmentation: Implement techniques like time-stretching, pitch-shifting, and adding white noise to improve model robustness.

For each project, encourage learners to go through a consistent cycle of building, evaluating, and refining their models. Promote thorough documentation of their findings, decisions, and any notable observations during exploratory data analysis and model evaluation. Additionally, encourage the sharing of findings, possibly through presentations or reports, to foster a collaborative learning environment.

Remember to apply the principles and techniques learned in previous sessions, approach challenges systematically, and don't hesitate to revisit previous materials or explore new resources as you embark on your project. We encourage you to discuss and collaborate with your peers, sharing insights and learning collectively. Happy learning and exploring!
