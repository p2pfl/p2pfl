#!/bin/bash
# Run MNIST example with node failure simulation
# node-0 will fail after round 2 and recover after round 4

DISABLE_RAY=1 python -m p2pfl.examples.mnist.mnist_with_failure \
  --nodes 4 \
  --rounds 6 \
  --epochs 1 \
  --protocol memory \
  --failure_round 2 \
  --recovery_round 4 \
  --failure_nodes node-0 \
  2>&1 | tail -100

