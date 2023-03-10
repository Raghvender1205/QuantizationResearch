# Quantization research
`Quantization` is the process of reducing the number of bits used to represent the weights and activations of a neural network. This can be done to reduce the memory footprint of the model, which can be useful for deploying the model on devices with limited memory or for reducing the amount of data that needs to be transmitted over the network.

1. There are several approaches to quantizing a model, including `uniform quantization`, `non-uniform quantization`, and `quantization-aware training`. Each method has its own trade-offs and may be more or less suitable for a given application.
2. Determine the range of the `weights` and `activations`: The range of the `weights` and `activations` will determine the number of bits needed to represent them. You can use a technique like `min-max scaling` or `dynamic range scaling` to determine the range of the weights and activations.
3. `Quantize` the weights and activations: Once you have determined the range of the weights and activations, you can use this range to quantize the weights and activations to the desired number of bits. This may involve rounding or truncating the weights and activations to the nearest representable value.
4. `Fine-tune` the quantized model: Quantization can introduce some loss of accuracy, so you may need to `fine-tune` the quantized model to regain some of the `lost accuracy`. This can be done using techniques like quantization-aware training or fine-tuning with a lower learning rate.

## Techniques
### MinMax Quantize
In this, we quantize weights and activations to `8 bits` using a simple `MinMax Scaling`. The weights and activations are first normalized to the range `[0 , 1]` and then rounded to the nearest representable value. Then, quantized weights and activations are `de-normalized` and `re-scaled` to the original range.

To run this use `minmax_quantize.py` script
```bash
python minmax_quantize.py
```
This script loads the `VisionTransformer` model and determines the range of weights and activations using `MinMax Scaling`. It then quantizes the weights and activations to `8 bits` and then saves the trained model.

1. Download Dataset manually if you want using `wget`
```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
```

```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
```
```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
```

2. Run the script `main.py` or use this `Kaggle Notebook`.

https://www.kaggle.com/code/raghvender/minmax-scaling-quantization-on-vit/notebook