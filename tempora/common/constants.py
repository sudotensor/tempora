DEVICES = ["cpu", "cuda"]

NOISE_CORRUPTIONS = ["gaussian_noise", "shot_noise", "impulse_noise"]
BLUR_CORRUPTIONS = ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"]
WEATHER_CORRUPTIONS = ["snow", "frost", "fog", "brightness", "contrast"]
DIGITAL_CORRUPTIONS = ["elastic_transform", "pixelate", "jpeg_compression"]

CORRUPTIONS = NOISE_CORRUPTIONS + BLUR_CORRUPTIONS + WEATHER_CORRUPTIONS + DIGITAL_CORRUPTIONS

# Remember to update the following when more methods or datasets are added!
METHODS = ["adabn", "basic", "eta", "lame", "neo", "predbn", "sar", "shot", "tent"]
DATASETS = ["cifar-10", "cifar-100", "imagenet"]
