# Tested on g++ version 7.3.0, TensorFlow version 1.12
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
for i in *.cc; do
  echo $i
  g++ -std=c++11 -shared $i -o ${i::-2}so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
done

