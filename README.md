<!-- # MUSE: Flexible Voiceprint Receptive Fields and Multi-Path Fusion Enhanced Taylor Transformer for U-Net-based Speech Enhancemen (Accepeted at Interspeech 2024)
### Zizhen Lin, Xiaoting Chen, Junyu Wang

**Abstract:** 
 Achieving a balance between lightweight design and high performance remains a challenging task for speech enhancement. In this paper, we introduce Multi-path Enhanced Taylor (MET) Transformer based U-net for Speech Enhancement (MUSE), a lightweight speech enhancement network built upon the U-net architecture. Our approach incorporates a novel Multi-path Enhanced Taylor (MET) Transformer block, which integrates Deformable Embedding (DE) to enable flexible receptive fields for voiceprints. The MET Transformer is uniquely designed to fuse Channel and Spatial Attention (CSA) branches, facilitating channel information exchange and addressing spatial attention deficits within the Taylor-Transformer framework. Through extensive experiments conducted on the VoiceBank+DEMAND dataset, we demonstrate that MUSE achieves competitive performance while significantly reducing both training and deployment costs, boasting a mere 0.51M parameters.

MUSE was accepted by Interspeech 2024. [arxiv](https://arxiv.org/pdf/2406.04589) -->
## Pre-requisites
1. Clone this repository.
2. Run the following command to create a conda environment named imse and install the required Python dependencies in the imse environment.
```
conda env create -f environment.yml
```
4. Download and extract the [VoiceBank-DEMAND-16k](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k). Use downsampling.py to process the dataset to format it for model training. Please read the code carefully to understand the four variables required to run downsampling.py: clean_train_path, noisy_train_path, clean_test_path, and resample_path. Adjust them to your project directory and then run the script.
```
python downsampling.py
```
5.Run make_file_list.py to obtain the training.txt and test.txt files for model training.
```
python make_file_list.py --train_clean_path "your train_clean_path" --train_noisy_path "your noisy_train_path" --test_clean_path "your clean_test_path" --test_noisy_path "your noisy_test_path"
```

## Training
For single GPU (Recommend), MUSE needs at least 8GB GPU memery.
```
python train.py --config config.json --input_training_file training.txt --input_validation_file test.txt
```


## Inference
1. First, use inference.py to generate the denoised audio files from your model.
```
python inference.py --input_clean_wavs_dir "your clean_test_path" --input_noisy_wavs_dir "your noisy_test_path" --output_dir "your generated_files" --checkpoint_file "your g_best_pesq_3.xx"
```
2. Use create_test_list.py to generate the test_final.txt file corresponding to "your generated_files", which will be used when running cal_metrics.py.
```
python create_test_list.py --test_dir_to_scan "your generated_files"
```
3. Use cal_metrics.py to calculate the PESQ, CSIG, CBAK, COVL, and STOI metrics for your best model.
```
python cal_metrics.py --clean_wav_dir "your clean_test_path" --noisy_wav_dir "your generated_files" --input_test_file "your test_final.txt"
```
You can also use the pretrained best checkpoint file we provide in `paper_result/g_best_pesq_3.373`.<br>



## Acknowledgements
We referred to [MP-SENet](https://github.com/yxlu-0102/MP-SENet), [MB-TaylorFormer](https://github.com/FVL2020/ICCV-2023-MB-TaylorFormer), [MUSE](https://github.com/huaidanquede/MUSE-Speech-Enhancement)
