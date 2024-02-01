Modelo  adapatado de https://github.com/tugstugi/pytorch-saltnet 

## Training

1. Ejecutar el script  `sh corre.sh` el scrip tiene el siguiente contenido 
    * `python train.py --vtf --pretrained imagenet --loss-on-center --batch-size 6     --optim adamw --learning-rate 8e-4 --lr-scheduler noam --basenet senet154 --    max-epochs 600 --data-fold fold2 --log-dir runs/fold2  --resume runs/fold2/checkpoints/last-checkpoint-fold2.pth`
    * Tener en cuenta que el script indica donde queremos que se guarde el modelo a entrenar entre otros parametros.


2. Ejecutar el tensorboar para ver el el entrenamiento (`tensoboard --logdir runs/`)
3. Para ver el resultado de la segmentación se puede correr el script(`python probarIOU.py` o ejecutar en Jupyter Notebook PruebaEnNotebook.ipynb ) 
    * Tener en cuenta que la dirección del modelo debe de cuadrar con lo descrito en el punto 1, este esta especificado en la linea 157 en el archivo probarIOU     model = models.load("runs/fold8/models/last-model-fold8.pth")

`
