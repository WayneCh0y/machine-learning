#!/bin/bash
set -e

cd "$(dirname "$0")"

cc -Wall -Wextra -O2 -o train src/*.c 
./train