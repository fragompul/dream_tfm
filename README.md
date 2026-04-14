# dream_tfm
D.R.E.A.M.: Arquitectura Multi-Agente para Trading basada en Fusión de Sentimiento y Volatilidad

## 🚀 Instalación y Configuración

### ⚠️ IMPORTANTE: Orden de Instalación (Soporte GPU)

Para garantizar que el proyecto utilice la **GPU (NVIDIA H100)** y no la versión de CPU por defecto, es **obligatorio** instalar PyTorch manualmente antes de procesar el archivo `requirements.txt`. 

Si se instala el archivo de requerimientos primero, librerías como `lightning` o `pytorch-forecasting` podrían instalar una versión genérica de PyTorch que no detectará tu hardware.

#### 1. Crear y activar el entorno (Conda recomendado)
```bash
conda create -n dream_tfm python=3.13
conda activate dream_tfm
