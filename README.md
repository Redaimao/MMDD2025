## MMDD 2025: The 1st Multimodal Deception Detection Competition @MM2025


-   **Training datasets:** [Real-life Deception Detection](https://public.websites.umich.edu/~zmohamed/resources.html) (Real-life Trial), [Bag-of-Lies](https://iab-rubric.org/index.php/bag-of-lies), and the [Miami University Deception Detection Database](https://sc.lib.miamioh.edu/handle/2374.MIA/6067) (MU3D).
-   **Stage 1 evaluation dataset:** [Box-of-Lies](https://web.eecs.umich.edu/~mihalcea/downloads.html#multimodalDialogDeception). [Finished!]
-   **Stage 2 dataset:** [Ongoing!!!]


### **Stage 2 Datasets Features Downloading:**

All participants must sign an agreement before accessing the datasets on their original platforms. Competition organizers will not provide raw data directly to participants. Instead, extracted OpenFace features, affect features from pretrained models, and Mel spectrograms and audio waves are provided [here](https://entuedu-my.sharepoint.com/:f:/g/personal/xiaobao001_e_ntu_edu_sg/EjSbeaVHhExIp7NJ5SClWZYBP4VaCMR6JJi-PjNijsUbqA?e=A1mYda) (pwd: MMDD2025). These features do not contain any identifiable information.

#### Note： If you do not have an outlook/onedrive account and encounter a downloading issue, please contact xiaobao001@e.ntu.edu.sg and let us know the email account you use to access this OneDrive. We will add your access to this folder.


#### Description:
  #### 1. DOLOS dataset:

We extracted the visual and audio features on DOLOS raw_audio and raw_video, same as the feature extraction in stage 1.
The visual features contain face frames, openface features(AUs and gaze),
affect features (by emonet, 5 emotions + 1 valence + 1 arousal); The audio features include mel spectrograms and audio wave (sr=16000).
More details can be found in "Load_DOLOS_face_audio_wave_openface_affect.py"


1). Balanced split (464 lie samples and 365 truth samples) is sampled.

train_balanced_l493_t492_GT.txt

and the extracted features:

train_balanced_l493_t492_GT.pkl

A sample of loading these files is provided in "Load_DOLOS_data.py"



- **DOLOS** is a gameshow based deception dataset. The DOLOS dataset can be downloaded from [ROSE Lab, NTU](https://rose1.ntu.edu.sg/dataset/DOLOS/). 

Please refer to [DOLOS and Code](https://github.com/NMS05/Audio-Visual-Deception-Detection-DOLOS-Dataset-and-Parameter-Efficient-Crossmodal-Learning/tree/main) to understand the full training process.


#### 2. MPDE dataset:

We extracted the visual and audio features from MDPE raw_audio and raw_video. The visual features contain face frames, openface features(AUs and gaze),
affect features (by emonet, 5 emotions + 1 valence + 1 arousal); The audio features include mel spectrograms and audio wave (sr=16000).
More details can be found in "Load_MDPE_face_audio_wave_openface_affect.py"

Based on the official annotation of the dataset, we used the labels from the interviewee as the correct ground_truth label.

The original labels have grade 1 to 5, where 1 represents that the subjects have definitely deceived successfully and 5 represents that I have definitely not deceived successfully.
Therefore, for the grade 1&2, we regard them as lies (labeled as 0) and for the grade 3,4&5, we regard them as truth (labeled as 1).
Note that MPDE is very unbalanced; we provide two kinds of train splits. Users can choose to use any or both of them for better performance.

1). Unbalanced split (2987 lie samples and 492 truth samples) is separated into three parts:

MDPE_train_unbalanced_1200_part1.txt

MDPE_train_unbalanced_1200_part2.txt

MDPE_train_unbalanced_1079_part3.txt

and the respective extracted features are:

MDPE_train_unbalanced_1200_part1.pkl

MDPE_train_unbalanced_1200_part2.pkl

MDPE_train_unbalanced_1079_part3.pkl


2). Balanced split (493 lie samples and 492 truth samples) are sampled from the unbalanced split to make the pos./neg. samples with nearly same amount.

MDPE_train_balanced_l493_t492_GT.txt

and the extracted features:

MDPE_train_balanced_l493_t492_GT.pkl


A sample of loading these files is provide in "Load_MDPE_data.py"


- **MDPE** can be found in [data](https://huggingface.co/datasets/MDPEdataset/MDPE_Dataset)

Participants are encouraged to download the original dataset and extract their features.




## Testing the fusion module
```bash
sh train_test_feature.sh
```

## Training with different modalities and fusion methods
Please check ```models/fusion_net.py``` for more details

## Environment
Please check the ```environment.yml ``` for details


## Please cite the paper if you find it useful
```
@inproceedings{guo2023audio,
  title={Audio-visual deception detection: Dolos dataset and parameter-efficient crossmodal learning},
  author={Guo, Xiaobao and Selvaraj, Nithish Muthuchamy and Yu, Zitong and Kong, Adams Wai-Kin and Shen, Bingquan and Kot, Alex},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={22135--22145},
  year={2023}
}

@article{guo2024benchmarking,
  title={Benchmarking Cross-Domain Audio-Visual Deception Detection},
  author={Guo, Xiaobao and Yu, Zitong and Selvaraj, Nithish Muthuchamy and Shen, Bingquan and Kong, Adams Wai-Kin and Kot, Alex C},
  journal={arXiv preprint arXiv:2405.06995},
  year={2024}
}

```



