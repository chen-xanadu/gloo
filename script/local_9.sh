#! /bin/bash
# launch 9 local processes that are separated into 3 groups,
# inject failure in rank-0 process

# (cd gloo/build/ && cmake .. -DBUILD_BENCHMARK=0)
(cd gloo/build/ && make)
mkdir -p ./example

rm -f example/*
rm -f /tmp/1* /tmp/3* /tmp/6* /tmp/8*

PREFIX=test1 SIZE=9 RANK=2 ./build/gloo/examples/example1 &> ./example/2 &
PREFIX=test1 SIZE=9 RANK=3 ./build/gloo/examples/example1 &> ./example/3 &
PREFIX=test1 SIZE=9 RANK=4 ./build/gloo/examples/example1 &> ./example/4 &
PREFIX=test1 SIZE=9 RANK=5 ./build/gloo/examples/example1 &> ./example/5 &
PREFIX=test1 SIZE=9 RANK=6 ./build/gloo/examples/example1 &> ./example/6 &
PREFIX=test1 SIZE=9 RANK=7 ./build/gloo/examples/example1 &> ./example/7 &
PREFIX=test1 SIZE=9 RANK=8 ./build/gloo/examples/example1 &> ./example/8 &
PREFIX=test1 SIZE=9 RANK=0 ./build/gloo/examples/example1 &> ./example/0 &
PREFIX=test1 SIZE=9 RANK=1 ./build/gloo/examples/example1 | tee ./example/1

sleep 1

killall exapmle1
