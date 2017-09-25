#!/bin/zsh
echo "srcdir: {$1}, dstdir: {$2}"
set srcdir = $1
set dstdir = $2
set n=1000;
for i in "${srcdir}"/*; do
  [ "$((N--))" = 0 ] && break
  cp -t "${dstdir}" -- "$i"
done
