# **Conteo de entrada y salida de vehiculos

---

# *Description
El sistema analiza grabaciones en formato .mp4 provenientes de una cámara IP. 
El enfoque del proyecto es simple y directo: usar un único ROI (Región de Interés) para identificar el movimiento de vehículos y su dirección, sin depender de otros elementos del entorno.

---

# *Objetivos
- Detectar vehículos dentro de un área específica del video (ROI).
- Determinar si un vehículo está entrando o saliendo.
- Registrar fecha y hora de cada evento.
- Generar estadísticas diarias de entradas y salidas.

# Flujo de trabajo
Se usaran archivos .mp4 de la camara (no en vivo para este proyecto)
Para detectar esto se tomara 1 criterio.
El Area solo registrara el cuadrante interior(Estacionamiento)
Si es hacia afuera entonces Salidas +=1
Si es hacia adentro entonces Entradas +=1

---

# Datos Importantes
Dia
Salidas y entradas totales del dia
Distribucion de entradas y salidas cada 1 horas 
(usar un color para entrada y otro para salidas)
15 Horas Diarias (6:00 - 21:00) (Distribuir en eje X)

---

Hardware:
PC con GPU
Camara IP