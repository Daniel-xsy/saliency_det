## Saliency Detection
This repo contains course project in *Huazhong University of Science and Technology. School of Artificial Intelligence and Automation*. [Visual cognitive engineering] course.

#### Dataset

MDvsFA Infrared Images Object Segmentation dataset, some of the images maybe corrupted, you can simply run the following commands to filter corrupted images and generate an `.pkl` filename list. (Or you can directly use [img_file_list.pkl](./img_file_list.pkl))

```bash
python preprocess.py
```

#### Train

```bash
bash train.sh
```

#### Eval

```bash
python eval.py
```

#### Reference

```
@inproceedings{wang2019miss,
  title={Miss detection vs. false alarm: Adversarial learning for small object segmentation in infrared images},
  author={Wang, Huan and Zhou, Luping and Wang, Lei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={8509--8518},
  year={2019}
}
```

