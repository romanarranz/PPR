# Instalacion CUDA en Ubuntu 16.04 LTS

## Preparando el entorno

En primer lugar tendremos que borrar cualquier existencia de paquetes `nvidia` que haya en nuestro sistema, asi como los repositorios que actualmente dan problema ya que
la clave pública que emiten está bajo SHA1 el cual ya no da mas soporte apt.

La última versión que soporta CUDA del compilador de C es gcc-4.9.2 así que tendremos que eliminarlo también.

```bash
$ sudo apt-get clean
$ sudo apt-get autoclean
$ sudo apt-get autoremove
$ sudo apt-get remove --purge nvidia*
```

## Descargando CUDA

Podemos descargarnoslo desde la propia web de nvidia [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
Es aconsejable instalar el archivo `.run` debido a que **NO** nos dará problemas de clave pública como los .deb por las razones comentadas anteriormente.

## Instalando paquetes necesarios

Como hemos desinstalado el compilador de C y C++ necesitaremos volver a instalarlo pero con la versión que da soporte CUDA es decir con la versión 4.9.2
Si usamos la utilidad de paquetes de ubuntu no tendremos éxito ya que instalará por defecto 4.9.3 que es superior a la que CUDA da soporte.
Así que nos descargamos la versión 4.9.2 de los repositorios de GCC, en mi caso he usado los de Francia [GCC-Download](http://fr.mirror.babylon.network/gcc/releases/gcc-4.9.2/gcc-4.9.2.tar.bz2)

```bash
$ bzip2 -dk gcc-4.9.2.tar.bz2
$ tar -xf gcc-4.9.2.tar
$ cd gcc-4.9.2
$ ./contrib/download_prerequisites
$ cd ..
$ mkdir objdir
$ cd objdir
$ $PWD/../gcc-4.9.2/configure --prefix=$HOME/gcc-4.9.2 --enable-languages=c,c++,fortran,go --disable-multilib
$ make
$ make install
```

Ahora necesitaremos deshabilitar el servidor X de nuestro equipo para comenzar la instalación

```bash
$ sudo service lightdm stop
```

Ya podemos proceder a instalar cuda

```bash
$ sudo chmod +x cuda.run
$ sudo
```

liblouisutdml-bin
shared-mime-info

liblouisutdml16
libxml2


## Documentación de CUDA

Obligatorios

- [CUDA Docs](http://docs.nvidia.com/cuda/index.html#axzz48jHKWvlH)
- [Compilador NVCC](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#axzz48jHKWvlH)
- [Tecnologías](https://developer.nvidia.com/key-technologies)
- [CUDA Education](https://developer.nvidia.com/cuda-education-training)
- [CUDA Additional Resources](https://developer.nvidia.com/additional-resources)

Opcionales
- [Librerias](https://developer.nvidia.com/gpu-accelerated-libraries)
- [OpenACC](https://developer.nvidia.com/openacc)
