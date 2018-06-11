alpine:
	docker build -t lablup/jail-hook-dev:alpine -f Dockerfile.alpine .
	docker run --rm -it -v $(shell pwd):/root lablup/jail-hook-dev:alpine /bin/sh -c 'cd /root; make inner'

debian:
	docker build -t lablup/jail-hook-dev:debian -f Dockerfile.debian .
	docker run --rm -it -v $(shell pwd):/root lablup/jail-hook-dev:debian /bin/sh -c 'cd /root; make inner'

inner:
	gcc -Wall -shared -fPIC -o libbaihook.so patch-cuda.c patch-libs.c -ldl
