## Etapa 1 - Inspección de datos

### Decisiones metodológicas

- Se normalizaron los píxeles a [0, 1].
- La estandarización se reserva para PCA y se ajustará solo con train.
- Se realizó split estratificado 80/20.

### Figuras generadas

- 15 imágenes aleatorias.
- Distribución de clases.
- Muestras agrupadas por clase.

### Tablas generadas

- Distribución de clases.
- Comparación de proporciones full/train/test.

### Texto candidato para informe

> Se realizó una inspección inicial del dataset Fashion-MNIST reducido, compuesto por imágenes de 28x28 píxeles en escala de grises. Las variables predictoras fueron separadas de las etiquetas, que se conservaron únicamente para análisis exploratorio, estratificación e interpretación posterior de los resultados. Los valores de píxel fueron normalizados al rango `[0, 1]`, mientras que la estandarización se reservó para la etapa de PCA. Finalmente, se aplicó una partición estratificada 80/20 para preservar la proporción de clases en los conjuntos de entrenamiento y evaluación.
>
> Para garantizar reproducibilidad y comparabilidad entre etapas, la partición estratificada preservó los índices originales de las muestras. Esto permitió reutilizar las mismas imágenes en comparaciones posteriores entre PCA, autoencoders determinísticos y beta-VAE. Además, las visualizaciones exploratorias y las tablas de distribución fueron exportadas para su inclusión directa en el informe.

## Etapa 2

> Para reducir la dimensionalidad se aplicó PCA implementado con `numpy` mediante descomposición en valores singulares. Previamente, los datos fueron estandarizados usando exclusivamente la media y el desvío del conjunto de entrenamiento, evitando leakage hacia el conjunto de evaluación. El número de componentes principales se seleccionó como el mínimo necesario para explicar al menos el 90% de la varianza acumulada. Las imágenes reconstruidas desde el espacio PCA permitieron evaluar visualmente la pérdida de información inducida por la reducción dimensional, preservándose principalmente la estructura global de las prendas y descartándose detalles de menor varianza.
>
> El Autoencoder determinístico se entrenó sobre las imágenes normalizadas al rango `[0, 1]`, utilizando como función de pérdida el error cuadrático medio entre la imagen original y su reconstrucción. La dimensión latente se fijó igual al número de componentes principales necesarias para explicar el 90% de la varianza en PCA, permitiendo una comparación más controlada entre una técnica lineal y una representación no lineal aprendida. La evaluación se realizó sobre el mismo conjunto utilizado para evaluar PCA, y las reconstrucciones visuales se compararon sobre las mismas diez imágenes seleccionadas con semilla fija. Las curvas de pérdida de entrenamiento y validación permiten analizar la convergencia del modelo y detectar posibles indicios de sobreajuste.
