#!/bin/bash
executable="./task04"
iterations=25
for i in $(seq 1 $iterations); do
  filename="test/${i}.png"
  outimage="binary/${i}.png"
  $executable "$filename" "$outimage"
done
