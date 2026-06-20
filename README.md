# Evaluación de trayectorias en robots mediante redes neuronales de grafos

## Objetivos del Proyecto
El propósito principal de este proyecto es el desarrollo de un pipeline automatizado de ingeniería de datos y el entrenamiento de un modelo de aprendizaje profundo hñibrido capaz de predecir la aceptabilidad social de trayectorias de navegación robótica.

Los objetivos específicos del repositorio incluyen:

* **Procesamiento y Modelado Topológico:** Transformar datos crudos de simulaciones en estrcutras de grafos (homogéneos y heterogéneos) utilizando PyTorch Geometric para capturar la geometría relacional del entorno.
* **Integración Contextual Multimodal:** Incorporar variables cualitativas y semánticas de alto nivel obtenidas mediante Modelos de Lenguaje Grande (LLM) para condicionar la evaluación del escenario.
* **Modelado Espacio-Temporal:** Implementar y entrenar una arquitectura híbrida basada en Redes Convolucionales de Grafos y Unidades Recurrentes para evaluar secuencias dinámicas de navegación.
*  **Estandarización y Robustez:** Aplicar técnicas de normalización orientadas a metas y aumento simétrico de datos para mitigar la escasez de muestras y estabilizar el aprendizaje supervisado.

## Características Principales del Framework

* **Configuración Dinámica:** Control absoluto del flujo de datos, hiperparámetros y flags de contexto a través de archivos centralizados YAML.
* **Abstracción Heterogénea:** Representación explícita y asimétrica de entidades del entorno: Robots, Humanos, Objetos, Paredes, Metas y un nodo Escenario virtual.
* **Preparación de Datos Eficiente:** Utilidade nativas para la ingesta de JSONs, normalización determinista, empaquetado por lotes dinámicos y duplicación de datos.
* **Arquitectura Modular Desacoplada:** Permite intercacmbiar fácilmente los bloques de la red sin tener que reescribir la lógica de entrenamiento ni de procesamiento de datos.
* **Control y Monitoreo de Experimentos:** Gestión automatizada de puntos de control para guardar los mejores pesos del modelo, fijación de semillas para garantizar la reproducibilidad y registro de logs para analizar la curva de convergencia de la pérdida.

## Realizado por
* **Guadalupe González Santos** – *Universidad de Extremadura (TFG)*
* Basado en la investigación de: A. Kapoor, S. Swamy, P. Bachiller y L. J. Manso.
