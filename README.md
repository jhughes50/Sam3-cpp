# SAM3-cpp 

This project use pytorch's jit compiler to export the individual models to be used in cpp. 
The model pre and post processing has been ported to cpp as well as tokenization for the text encoder. 

We also convert to `bFloat16` on export to speed up the model. With this we can run inference at under 0.5 seconds on
an RTX 4000 ada. 

The model uses around 2.5gb VRAM when quantized and around 5 without quantization. 

## Upcoming Work
1. Currently we can only do single class segmentation. Multi-class support is coming.
2. We did not migrate the prompt encoder, I don't really need it and it was being troublesome but if someone wants to tackle that, I'm happy to include it.
3. Remove AI slop, I mostly let claude run on this to see if it was even possible. There's a lot of inefficiencies that need to be cleaned up.


