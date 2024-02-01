
#rm -r runs/
#mkdir runs

#python train.py --vtf --pretrained imagenet --loss-on-center --batch-size 1 --optim adamw --learning-rate 5e-4 --lr-scheduler noam --basenet resnet152 --max-epochs 16 --data-fold fold9 --log-dir runs/fold9  --resume runs/fold9/checkpoints/last-checkpoint-fold9.pth 
python train.py --vtf --pretrained imagenet --loss-on-center --batch-size 6 --optim adamw --learning-rate 8e-4 --lr-scheduler noam --basenet senet154 --max-epochs 600 --data-fold fold2 --log-dir runs/fold2  --resume runs/fold2/checkpoints/last-checkpoint-fold2.pth 
