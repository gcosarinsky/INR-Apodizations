-El chat será en castellano, pero el código debe estar en inglés. Esto es para mantener consistencia con el resto del proyecto y facilitar la colaboración con otros desarrolladores.

-Los parámetros que se pasan a los scripts deben ser pasados mediante json o yaml, no hardcodeados dentro del script. Esto hace que el código sea más flexible y reutilizable, y permite cambiar los parámetros sin modificar el código.

-Agregar docstrings a funciones y clases para explicar su propósito, parámetros de entrada, valores de retorno y cualquier excepción que puedan lanzar. Esto mejora la documentación del código y facilita su comprensión por parte de otros desarrolladores. Esto debe hacerse en inglés.

- Al crear scripts para tests iniciales (usualmente en la carpeta `sandbox`) no crear una lógica compleja; mantenerlos simples y enfocados en verificar funcionalidades específicas. Por ejemplo, no crear una funcion main() ni usar argparse para manejar argumentos de línea de comandos.