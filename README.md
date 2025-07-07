# lab9
lab 9 machine learning

Integrantes ( grupo 1 ) 

Xiomara Mayela
Diego Antonio Rosario Palomino
Coralie Figueroa

Comentario del enfoque:
Para abordar la tarea de clasificación de imágenes con CNNs, decidimos reemplazar algunas capas convolucionales estándar por deformable convolutions nativas de PyTorch. Esto nos permitió mejorar la capacidad del modelo para adaptarse a variaciones geométricas en los datos, sin aumentar significativamente el número de parámetros. Entrenamos el modelo con Tiny ImageNet, utilizando técnicas de normalización y aumento de datos para optimizar el rendimiento.

--

Gracias a las deformable convolutions y al activation dropout (llamado incorrectamente en el código como excitation dropout), se logra una buena performance en un dataset limitado en tamaño como Tiny ImageNet, incluso sin necesidad de aplicar técnicas de data augmentation.

Las deformable convolutions permiten a la red adaptar su campo receptivo dinámicamente, mejorando la capacidad del modelo para generalizar en contextos con variaciones estructurales en las imágenes.

Por otro lado, el uso de dropout aplicado directamente en las activaciones de las fully connected layers mejora la generalización del clasificador sin afectar la capacidad de aprendizaje del extractor de características.

En conjunto, estas técnicas permiten entrenar modelos robustos y eficientes con pocos datos, evitando el sobreajuste sin depender de aumentos artificiales.