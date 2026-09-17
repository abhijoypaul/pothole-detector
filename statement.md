# Problem Statement
Road surface damage, particularly potholes, causes significant vehicle wear, accidents, and maintenance costs. Manual inspection is slow, subjective, and difficult to scale. We need an automated computer vision solution to detect potholes in road images and quantify their severity, providing a structured assessment without relying on computationally expensive deep learning models.

## Scope of the Project
- **In-Scope**: Detecting potholes in static 2D road images using classical computer vision techniques (Thresholding and Graph-Cut/MRF). Generating severity reports (Low/Medium/High) based on damage area, producing visual overlay masks, and evaluating against Pascal VOC bounding-box annotations.
- **Out-of-Scope**: Video processing, real-time edge deployment, tracking potholes across multiple frames, 3D depth estimation, and deep-learning based instance segmentation.

## Target Users
- **Municipalities & City Planners**: To automatically assess road conditions from survey photos and prioritize repair budgets.
- **Road Maintenance Crews**: To quickly identify and locate severe damage areas.
- **Computer Vision Students/Researchers**: As an educational baseline demonstrating classical Markov Random Field (MRF) energy minimization and adaptive thresholding.

## High-level Features
- **Dual Segmentation Methods**: Adaptive Thresholding (fast morphological approach) and Graph-Cut MRF (robust spatial-coherence approach).
- **Severity Scoring**: Calculates the damage area ratio and categorizes damage as Low (<2%), Medium (2-8%), or High (>8%).
- **Automated Reporting**: Generates comprehensive batch summaries in both CSV and JSON formats for downstream analytics.
- **Visual Overlays**: Generates images with red masks superimposed over detected potholes for manual verification.
- **Intersection over Union (IoU) Evaluation**: Automatically parses Pascal VOC XML annotations and computes IoU metrics to validate detection accuracy.
