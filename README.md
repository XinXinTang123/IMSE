# IMSE: Efficient U-Net-based Speech Enhancement using Inception Depthwise Convolution and Amplitude-Aware Linear Attention
### Xinxin Tang Bin Qin Yufang Li


**Abstract:** 
 Achieving a balance between lightweight design and high performance remains a significant challenge for speech enhancement (SE) tasks on resource-constrained devices. Existing state-of-the-art methods, such as MUSE, have established a strong baseline with only 0.51M parameters by introducing a Multi-path Enhanced Taylor (MET) transformer and Deformable Embedding (DE). However, an in-depth analysis reveals that MUSE still suffers from efficiency bottlenecks: the MET module relies on a complex ”approximatecompensate” mechanism to mitigate the limitations of Taylorexpansion-based attention, while the offset calculation for deformable embedding introduces additional computational burden. This paper proposes IMSE, a systematically optimized and ultra-lightweight network. We introduce two core innovations: 1) Replacing the MET module with Amplitude-Aware Linear Attention (MALA). MALA fundamentally rectifies the ”amplitude-ignoring” problem in linear attention by explicitly preserving the norm information of query vectors in the attention calculation, achieving efficient global modeling without an auxiliary compensation branch. 2) Replacing the DE module with Inception Depthwise Convolution (IDConv). IDConv borrows the Inception concept, decomposing large-kernel operations into efficient parallel branches (square, horizontal, and vertical strips), thereby capturing spectrogram features with extremely low parameter redundancy. Extensive experiments on the VoiceBank+DEMAND dataset demonstrate that, compared to the MUSE baseline, IMSE significantly reduces the parameter count by 16.8% (from 0.513M to 0.427M) while achieving competitive performance comparable to the state-of-the-art on the PESQ metric (3.373). This study sets a new benchmark for the trade-off between model size and speech quality in ultralightweight speech enhancement

<!--   MUSE was accepted by Interspeech 2024. [arxiv](https://arxiv.org/pdf/2406.04589)       -->
## Pre-requisites
1. Clone this repository.
2. Run the following command to create a conda environment named imse and install the required Python dependencies in the imse environment.
```
conda env create -f environment.yml
```
3. activate conda env

```
conda activate imse
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
