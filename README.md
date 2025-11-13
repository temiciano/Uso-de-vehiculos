Objetivos:
Detectar cuantos autos entran y salen de un estacionamiento
Detectar cuando la reja se abre
Detectar si un auto entra o sale
Hora/Dia de cada salida y entrada

Pasos:
Se usaran archivos .mp4 de la camara (no en vivo para este proyecto)
Para detectar esto se tomara 1 criterio.
Habra siempre un sensor siguiendo el letrero del porton (10KM), mientras este letrero entre en X Area la camara registrara el movimiento en la segunda area.
La segunda Area corresponde al espacio en el que los autos entran y salen.
La segunda Area solo registrara el cuadrante interior(Estacionamiento), al momento de abrirse la reja se detecta el movimiento del vehiculo.
Si es hacia afuera entonces Salidas +=1
Si es hacia adentro entonces Entradas +=1


Hardware:
PC con GPU
Camara IP