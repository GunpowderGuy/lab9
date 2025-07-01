# lab9
lab 9 machine learning

Integrantes ( grupo 1 ) 

Xiomara Mayela
Diego Antonio Rosario Palomino
Coralie Figueroa

Comentario del enfoque:
Para abordar la tarea de clasificación de imágenes con CNNs, decidimos reemplazar algunas capas convolucionales estándar por deformable convolutions nativas de PyTorch. Esto nos permitió mejorar la capacidad del modelo para adaptarse a variaciones geométricas en los datos, sin aumentar significativamente el número de parámetros. Entrenamos el modelo con Tiny ImageNet, utilizando técnicas de normalización y aumento de datos para optimizar el rendimiento.