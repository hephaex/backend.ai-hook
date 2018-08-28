# Backend.AI Hook

A private hook library to override library-level functions inside REPL containers


## Features

 * Override `sysconf()` to expose the number of actually schedulable CPU cores based on sysfs cgroup
   CPU set information.
<<<<<<< HEAD
 * Override `scanf()` to get the user keyboard input via the local Backend.AI-Agent.
=======
 * Override `scanf()` to get the user keyboard input via the local Backend.AI Agent.
>>>>>>> 40aeb22d20a28388cd0a3df91383b9a73120916e

## How to build

For musl-compatible build using Alpine Linux, run:

```sh
make alpine
```

For glibc-compatible build using Debian Linux, run:

```sh
make debian
```
