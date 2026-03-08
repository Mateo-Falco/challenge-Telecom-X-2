# TelecomX — Parte 2: Modelos Predictivos de Churn

Continuacion del proyecto de analisis de churn de TelecomX. Esta segunda parte toma el dataset limpio generado en la Parte 1 y construye un pipeline completo de machine learning para predecir la cancelacion de clientes, incluyendo preprocesamiento, entrenamiento de cuatro modelos, evaluacion comparativa y analisis de variables relevantes.

---

## Descripcion del problema

Con los datos ya limpios y estandarizados, el objetivo es construir modelos capaces de identificar que clientes tienen mayor probabilidad de cancelar el servicio. Esto permite al area comercial intervenir de forma proactiva antes de que se produzca la baja.

---

## Estructura del repositorio

```
├── TelecomX_Parte2.ipynb     # Notebook principal de la parte 2
├── TelecomX_limpio.csv       # Dataset tratado generado en la Parte 1
└── README.md
```

---

## Pipeline del notebook

### 1. Carga del dataset
Se carga el CSV exportado al final de la Parte 1, que contiene los datos ya limpios, estandarizados y con variables en español.

### 2. Eliminacion de columnas sin valor predictivo
Se eliminan tres columnas antes del modelado:
- `id_cliente`: identificador unico sin informacion predictiva
- `cargo_total`: variable derivada (`cargo_mensual` × `meses_contratado`), introduce multicolinealidad
- `cargo_diario`: variable derivada (`cargo_mensual` / 30), duplica la informacion de `cargo_mensual`

### 3. Codificacion de variables categoricas
Se aplica One-Hot Encoding sobre las columnas de tipo object (genero, tipo de contrato, metodo de pago, servicio de internet) para hacerlas compatibles con los algoritmos de machine learning.

### 4. Analisis de desbalance de clases
Se verifica la distribucion de la variable objetivo. El dataset presenta un desbalance de aproximadamente 74% activos / 26% cancelados, lo que puede sesgar los modelos hacia la clase mayoritaria.

### 5. Balanceo con SMOTE
Se aplica SMOTE (Synthetic Minority Oversampling Technique) exclusivamente sobre el conjunto de entrenamiento para generar ejemplos sinteticos de la clase minoritaria y equilibrar las clases sin filtrar datos reales del conjunto de prueba.

### 6. Normalizacion
Se aplica `StandardScaler` para los modelos sensibles a la escala (Regresion Logistica, KNN, SVM). Random Forest no requiere normalizacion. El scaler se ajusta solo sobre los datos de entrenamiento y se aplica al conjunto de prueba.

### 7. Analisis exploratorio complementario
- Matriz de correlacion de variables numericas clave
- Boxplots y scatter plot de tiempo de contrato y gasto total vs cancelacion

### 8. Entrenamiento de modelos
Se entrenan cuatro modelos con una division 80/20 estratificada:

| Modelo | Normalizacion | Motivo |
|---|---|---|
| Regresion Logistica | Si | Sensible a la escala; optimiza coeficientes por gradiente |
| KNN (k=11) | Si | Basado en distancias euclideas |
| Random Forest | No | Basado en umbrales de corte, no en magnitudes |
| SVM (kernel RBF) | Si | Sensible a la escala en el calculo del kernel |

### 9. Evaluacion
Cada modelo se evalua con exactitud, precision, recall, F1-score y matriz de confusion. Se genera una tabla comparativa y un grafico de barras con las cuatro metricas por modelo.

### 10. Analisis de variables relevantes
- **Regresion Logistica**: coeficientes por variable (efecto protector vs efecto de riesgo)
- **KNN**: permutation importance (caida en F1 al permutar cada variable)
- **Random Forest**: importancia por reduccion de impureza de Gini
- **SVM**: coeficientes de un SVM lineal entrenado en paralelo para interpretabilidad
- Grafico comparativo con importancias normalizadas 0-1 entre los cuatro modelos

### 11. Informe final
Markdown estructurado con los principales factores de cancelacion identificados, comparacion critica de modelos y seis estrategias de retencion basadas en los resultados.

---

## Principales factores de cancelacion identificados

- Contrato mes a mes (mayor riesgo en todos los modelos)
- Poca antiguedad (primeros 12 meses son el periodo critico)
- Servicio de internet por fibra optica
- Cargo mensual elevado
- Metodo de pago por cheque electronico
- Ausencia de servicios adicionales contratados

---

## Tecnologias utilizadas

- Python 3
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- imbalanced-learn (SMOTE)

---

## Requisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

---

## Como ejecutar

1. Asegurarse de tener el archivo `TelecomX_limpio.csv` en el mismo directorio (generado al ejecutar la Parte 1).
2. Abrir el notebook:
```bash
jupyter notebook TelecomX_Parte2.ipynb
```
3. Ejecutar todas las celdas en orden.

---

## Parte 1

El analisis exploratorio, limpieza de datos y generacion del CSV se encuentran en el repositorio de la Parte 1: [TelecomX — Analisis de Churn](https://github.com/tu-usuario/telecomx-parte1)
