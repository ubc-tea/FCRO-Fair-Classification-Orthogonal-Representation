This repository contains the PyTorch implementation of our [IPMI 2023](https://www.ipmi2023.org/en/) paper ["On Fairness of Medical Image Classification with Multiple Sensitive Attributes via Learning Orthogonal Representations"](https://arxiv.org/abs/2301.01481).

Wenlong Deng*, Yuan Zhong*, [Qi Dou](https://www.cse.cuhk.edu.hk/~qdou/), [Xiaoxiao Li](https://xxlya.github.io/xiaoxiao/)

[[Paper]](https://arxiv.org/abs/2301.01481)

# Usage

### Setup

### Datasets

Please download the original CheXpert dataset [here](https://stanfordmlgroup.github.io/competitions/chexpert/), and supplementary demographic data [here](https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf).

In our paper, we use an augmented version of CheXpert. Please download the metadata of the augmented dataset [here](https://drive.google.com/file/d/1ggyr0VLsXCAGK7yGzkNaHTVIu0Qk0kOk/view?usp=share_link), and put it under the `./metadata/` directory.

### Pretrained Models

Please download our pretrained models using 5-fold cross validation [here](https://drive.google.com/file/d/1DzOKziWvBoxfdYW_Q5sAUz2l-4ZDGwmu/view?usp=share_link), and put them under the `./checkpoint/` directory.

### Run a single experiment

```bash
python train.py --image_path [image_path] --exp_path [exp_path] --metadata [metadata] --lr [lr] --weight_decay [weight_decay] --epoch [epoch] --batch_size [batch_size] -a [sensitive_attributes] --dim_rep [dim_rep] -wc [wc] -wr [wr] --subspace_thre [subspace_thre] -f [fold] --cond --moving_base --from_sketch
```

For more information, please execute `python train.py -h` for help.

Here is an example of how to run a experiment on fold 0 from sketch:

```bash
# Train from sketch, i.e., train the sensitive head first, then train the target head.
python train.py --image_path XXX -f 0 --cond --from_sketch
```

Here is another example of how to train the target model using a pretrained sensitive model:

```bash
python train.py --image_path XXX -f 0 --cond
```

By default, the pretrained sensitive model under the `./checkpoint/` directory will be used. If you want to customize it, please use `--pretrained_path` option.

To calculate column orthogonal loss using accumulative space construction variant, please use --`moving_space` option.

### Test

After installing our pretrained model and metadata, you can reproduce our 5-fold cross validation results in our paper by running:

```bash
# Running test using model of fold 0. Please run full 5-fold to reproduce our results
python train.py --test --image_path XXX -f 0
```

You may customize `--pretrained_path` and `--sensitive_attributes` commands to use other pretrained models or test on other sensitive attributes combinations.

## Citation

If you find this work helpful, feel free to cite our paper as follows:
