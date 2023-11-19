This is the official repository for our paper, "Multiple Toddler Tracking in Indoor Videos" accpted for publication in CV4Smalls Workshop at WACV 2024.

# MTTSort: Enhanced DeepSort for Toddler Tracking

<p align="center">
    <img src="figures/MTTSort_logo.webp" alt="MTTSort Logo"/>
</p>

MTTSort is an enhanced version of the DeepSort algorithm, specifically tailored for tracking toddlers in various environments. By leveraging advanced deep learning techniques, MTTSort provides accurate and real-time tracking, making it an ideal solution for monitoring and ensuring the safety of toddlers.

## Features

- Real-time toddler tracking with high accuracy.
- Advanced deep learning techniques for improved performance.
- Customizable tracking parameters for different environments.
- Integration with YOLOv8 for robust object detection.
- Genetic Algorithm Evolution for fine-grained hyper-parameters tuning.

## Getting Started

Follow these steps to set up and run MTTSort on your local machine.

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.8 or higher
- Pip package manager

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/MTTSort.git
    ```

2. Navigate to the cloned directory:

    ```bash
    cd MTTSort
    ```

3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

To use MTTSort for toddler tracking, follow these steps:

1. Set the working directory:

    ```bash
    cd path/to/detect/folder
    ```

2. Run the genetic algorithm script for evolution:

    ```bash
    python evolve.py
    ```

3. Run the detection model with HOTA and MOTA and IDF1 scores calculation:

    ```bash
    python predict_evaluate.py
    ```

4. Run only model for detections and tracks:

    ```bash
    python predict_frames.py
    ```

### Data Preperation

For annotation and labeling of images used in testing the MTTSort model, [LabelMe](https://github.com/wkentaro/labelme) was utilized. LabelMe is an open-source graphical image annotation tool written in Python.

After preparing the JSON files from Labelme run the json_to_text.py script to transform the labels to the desired format. 

1. Run to transform the json labels to text:

    ```bash
    python json_to_text.py
    ```

### Evaluation

The data used for evaluation should be in such format 
```bash
    frame_id, Subject_ID, x_top, y_top, x_bottom, y_bottom
```

### Results


Here's a demonstration of the MTTSort algorithm in action:

<p align="center">
    <img src="figures/MTTSort_demo.gif" alt="MTTSort Demo"/>
</p>
### Support and Contributions

For support, questions, or contributions, please open an issue or submit a pull request in the repository.

## Citation

If you find use this code or dataset for your research, please consider citing our paper:
```
@inproceedings{Amraee2024toddlertracking,
  title={Multiple Toddler Tracking in Indoor Videos},
  author={Amraee, Somaieh and Galoaa, Bishoy  and Goodwin, Matthew  and  Hatamimajoumerd,  Elaheh  and Ostadabbas, Sarah},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW)},
  month={1},
  year={2024}
  }
```

### License

This project is licensed under the [Your License Name]. For more information, see the `LICENSE` file in the repository.

### Acknowledgements

Special thanks to the original DeepSort and YOLOv8 contributors for their foundational work.
