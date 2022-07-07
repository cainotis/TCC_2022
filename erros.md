## 29 de junho

- [libGL error: failed to load drivers iris and swrast in Ubuntu 20.04](https://askubuntu.com/questions/1352158/libgl-error-failed-to-load-drivers-iris-and-swrast-in-ubuntu-20-04)

- ImportError: libpython3.8.so.1.0 -> change to python3.10

- [AttributeError: module 'collections' has no attribute 'Container'](https://stackoverflow.com/questions/69468128/fail-attributeerror-module-collections-has-no-attribute-container)

 -> editar /home/$user/anaconda3/envs/$name/lib/python3.10/site-packages/contracts/library/miscellaneous_aliases.py 
 import collections.abc as collections


- NotImplementedError: numpy() is only available when eager execution is enabled.