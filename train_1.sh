python train.py \
    ~/Public/share/Data/Cifar10 \
    ~/Documents/Models/bbdm_new/ddpm_test_cifar10_100.pth \
    -b 16 \
    -d cifar10 \
    -e 250 \
    -t 1000 \
    --beta_range 0.001 0.029 \
    -exp ddpm_cifar10_100.exp \
    --show_verbose \
    --device cuda:1
