# PSFR-GAN in PyTorch 

> We only provide test codes at this time. 

[Progressive Semantic-Aware Style Transformation for Blind Face Restoration]()  
[Chaofeng Chen](https://chaofengc.github.io), [Xiaoming Li](https://csxmli2016.github.io/), [Lingbo Yang](https://lotayou.github.io), Xianhui Lin, [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang/), [Kwan-Yee K. Wong](https://i.cs.hku.hk/~kykwong/)

![](test_dir/test_hzgg.jpg)
![](test_hzgg_results/hq_final.jpg)

## Getting Started

### Prerequisites and Installation
- Ubuntu 18.04
- CUDA 10.1  
- Clone this repository
    ```
    git clone https://github.com/chaofengc/PSFR-GAN.git
    cd PSFR-GAN
    ```
- Python 3.7, install required packages by `pip3 install -r requirements.txt`  

### Download Pretrain Models and Dataset
Download the pretrained models from the following link and put them to `./pretrain_models`  
- [GoogleDrive](https://drive.google.com/drive/folders/1Ubejhxd2xd4fxGc_M_LWl3Ux6CgQd9rP?usp=sharing)
- [BaiduNetDisk](https://pan.baidu.com/s/1_5MzYnhkUOrV35A_sBKulw), extract code: `4uip`

### Test single image
Run the following script to enhance face(s) in single input  
```
python test_enhance_single_unalign.py --test_img_path ./test_dir/test_hzgg.jpg --results_dir test_hzgg_results --gpus 1
```

This script do the following things:
- Crop and align all the faces from input image, stored at `results_dir/LQ_faces`  
- Parse these faces and then enhance them, results stored at `results_dir/ParseMaps` and `results_dir/HQ`  
- Paste then enhanced faces back to the original image `results_dir/hq_final.jpg`  
- You can use `--gpus` to specify how many GPUs to use, `<=0` means running on CPU. The program will use GPU with the most available memory. Set `CUDA_VISIBLE_DEVICE` to specify the GPU if you do not want automatic GPU selection.  

### Test image folder 
To test multiple images, we first crop out all the faces and align them use the following script.  
```
python align_and_crop_dir.py --src_dir test_dir --results_dir test_dir_align_results
```  

For images (*e.g.* `multiface_test.jpg`) contain multiple faces, the aligned faces will be stored as `multiface_test_{face_index}.jpg`  
And then parse the aligned faces and enhance them with  
```
python test_enhance_dir_align.py --dataroot test_dir_align_results --results_dir test_dir_enhance_results
```  
Results will be saved to three folders respectively: `results_dir/lq`, `results_dir/parse`, `results_dir/hq`.   
*Note: This is used to test a large amounts of data, and we do not paste the faces back.*

## Citation
```
@InProceedings{ChenPSFRGAN,
    author = {Chen, Chaofeng and Li, Xiaoming and Lin, Xianhui and Lingbo, Yang and Zhang, Lei and Wong, KKY},
    title = {Progressive Semantic-Aware Style Transformation for Blind Face Restoration},
    year = {2020}
}
```

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## Acknowledgement

This work is inspired by [SPADE](https://github.com/NVlabs/SPADE), and closed related to [DFDNet](https://github.com/csxmli2016/DFDNet) and [HiFaceGAN](https://github.com/Lotayou/Face-Renovation). Our codes largely benefit from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
