## MMDD 2025: The 1st Multimodal Deception Detection Competition @MM2025



### **Datasets**

All participants must sign an agreement before accessing the datasets on their original platforms. Competition organizers will not provide raw data directly to participants. Instead, extracted OpenFace features, affect features from pretrained models, and Mel spectrograms (generated using PyTorch) are provided [here](https://entuedu-my.sharepoint.com/personal/xiaobao001_e_ntu_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxiaobao001%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FResearch%2FMMDD%5Ffeatures&ga=1) (pwd: MMDD123456). These features do not contain any identifiable information.

-   **Training datasets:** [Real-life Deception Detection](https://public.websites.umich.edu/~zmohamed/resources.html) (Real-life Trial), [Bag-of-Lies](https://iab-rubric.org/index.php/bag-of-lies), and the [Miami University Deception Detection Database](https://sc.lib.miamioh.edu/handle/2374.MIA/6067) (MU3D).
-   **Stage 1 evaluation dataset:** [Box-of-Lies](https://web.eecs.umich.edu/~mihalcea/downloads.html#multimodalDialogDeception).
-   **Stage 2 testing dataset:** will be released on 9th May, 2025


- **DOLOS** is a gameshow based deception dataset. The DOLOS dataset can be downloaded from [ROSE Lab, NTU](https://rose1.ntu.edu.sg/dataset/DOLOS/). 

Please refer to [DOLOS and Code](https://github.com/NMS05/Audio-Visual-Deception-Detection-DOLOS-Dataset-and-Parameter-Efficient-Crossmodal-Learning/tree/main) to understand the full training process.

Participants are encouraged to download all the datasets as early as possible !!

## Testing the fusion module
```bash
sh train_test_features.sh
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
```


