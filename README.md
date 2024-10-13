
# Entrenamiento de Robots Autónomos Usando World Models en GTA V

## Descripción General del Proyecto

Este proyecto se centra en desarrollar un sistema basado en inteligencia artificial para entrenar a un robot a caminar de manera autónoma en entornos virtuales complejos utilizando tecnologías avanzadas como **World Models**, **Aprendizaje por Refuerzo Profundo** y **Transformers**. La plataforma de simulación utilizada es **Grand Theft Auto V (GTA V)**, que nos permite crear escenarios de entrenamiento realistas para el robot, mejorando sus habilidades de navegación y toma de decisiones en terrenos desafiantes.

### Objetivo Principal
El objetivo principal del proyecto es desarrollar un robot capaz de navegar y adaptarse de manera autónoma a diferentes tipos de terreno utilizando técnicas de IA, minimizando errores como problemas de equilibrio o colisiones.

## Introducción

Este sistema de IA está diseñado para cerrar la brecha tecnológica en Perú, donde la inversión en estas áreas es limitada. Al utilizar **World Models**, el robot aprende a interactuar eficientemente con su entorno a través de simulaciones. **GTA V** sirve como plataforma de simulación para generar datasets variados y realistas, lo que permite entrenar al robot en un entorno seguro y controlado.

---

## Descripción del Problema

En Perú, la adopción de sistemas autónomos y tecnologías de IA en sectores como la **minería**, **agricultura** y **transporte** aún es limitada. Los principales desafíos para el desarrollo de la robótica en el país incluyen:
- Falta de acceso a datos del mundo real.
- Altos costos de pruebas y desarrollo en entornos reales.
- Poca inversión en investigación de IA.

Este proyecto aborda estos desafíos mediante el uso de entornos simulados en **GTA V**, proporcionando una plataforma completa para entrenar a un robot a navegar de manera autónoma en diversos terrenos complejos e inspirados en el mundo real.

---

## Conjunto de Datos

El dataset utilizado para entrenar al robot se genera a partir de simulaciones realistas en **GTA V**. Las características clave del conjunto de datos incluyen:
- **Imágenes del entorno (RGB y 3D)**: Captura visual detallada de distintos tipos de terrenos, obstáculos y condiciones climáticas.
- **Acciones realizadas por el robot**, como avanzar, equilibrarse, girar y evitar obstáculos.
- **Recompensas y penalizaciones** basadas en el rendimiento del robot en cada escenario, para ayudar a optimizar su comportamiento.

### Ejemplo de Acciones y Recompensas

| Entorno                            | Acción del Robot                                      | Recompensa | Penalización       |
|-------------------------------------|------------------------------------------------------|------------|--------------------|
| Terreno plano con obstáculos bajos  | Paso adelante (0.3 m/s)                              | +5         | -1 (choque)        |
| Terreno resbaladizo                 | Paso cuidadoso (0.2 m/s)                             | +6         | -4 (caída)         |
| Terreno con pendientes moderadas    | Paso con ajuste de balance (0.4 m/s)                 | +9         | -2 (pérdida de equilibrio) |
| Vegetación densa                    | Avanzar lentamente y esquivar obstáculos (0.3 m/s)   | +7         | -1 (rozadura)      |

---

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

- **controller.py**: Gestiona la interacción del robot con el entorno, procesa las observaciones del entorno mediante el **VAE**, y elige acciones a través de la red neuronal **ActionNetwork**.
- **vae.py**: Implementa el **VAE** para aprender un espacio latente comprimido a partir de las imágenes del entorno.
- **rn.py**: Red neuronal simple que asigna las representaciones latentes del **VAE** a probabilidades de acciones.
- **utils.py**: Carga el conjunto de datos (imágenes) para el entrenamiento del VAE y facilita la preprocesamiento de datos.

---

## Ejecución del Proyecto

### Requisitos

Asegúrate de tener instaladas las siguientes librerías:
- `torch`
- `torchvision`
- `PIL` para procesamiento de imágenes
- Simulador del entorno, como **GTA V**

### Pasos para Ejecutar

1. Clona el repositorio y navega al directorio del proyecto:
    ```bash
    git clone https://tu-repositorio.git
    cd project
    ```

2. Entrena el modelo **Variational Autoencoder (VAE)** ejecutando:
    ```bash
    python vae.py
    ```

3. Inicia el controlador e interactúa con el entorno:
    ```bash
    python controller.py
    ```

---

## Resultados y Conclusión


---

## Colaboradores

- **Diego Eduardo Ballón Villar** (U201520327)
- **Christian Aaron Velasquez Borasino** (U202218075)
- **Joaquin Mauricio Eguia Castilla** (U202213539)
