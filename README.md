# Backend.AI Hook

A private hook library to override library-level functions inside REPL containers


## Features

 * Override `sysconf()` to expose the number of actually schedulable CPU cores based on sysfs cgroup
   CPU set information.
 * Override `scanf()` to get the user keyboard input via the local Backend.AI-Agent.

## How to build

For musl-compatible build using Alpine Linux, run:

```sh
make alpine
```

For glibc-compatible build using Debian Linux, run:

```sh
make debian
```
