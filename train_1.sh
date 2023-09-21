python train.py \
    ~/Public/share/Data/Cifar10 \
    ~/Documents/Models/bbdm_new/ddpm_cifar10.pth \
    -b 256 \
    -d cifar10 \
    -e 400 \
    -t 1000 \
    --beta_range 0.001 0.029 \
    -exp ddpm_cifar10.exp \
    --show_verbose \
    --device cuda:1 \
    --replace_experiment
