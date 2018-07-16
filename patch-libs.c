#include "patch.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#include <errno.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>

static const char *input_host = "127.0.0.1";
static const int input_port = 65000;


static int nproc_from_sysfs_cpuset()
{
    const size_t maxlen = 512;
    char line[maxlen];
    FILE *fin = fopen("/sys/fs/cgroup/cpuset/cpuset.cpus", "r");
    if (fin == NULL)
        return 0;

    if (fgets(line, maxlen, fin) == NULL)
        return 0;

    int result = 0;
    char *l = &line[0];
    char *ends = (l + (uintptr_t) strnlen(line, maxlen));
    do {
        char *endp;
        unsigned long int n = strtoul(l, &endp, 10);
        if (l == endp) {
            result = 0;
            break;
        }
        unsigned long int m = n;
        if (*endp == '-') {
            l = endp + 1;
            m = strtoul(l, &endp, 10);
            if (l == endp) {
                result = 0;
                break;
            }
        }
        result += (m - n + 1);
        l = endp;
        while (l < ends && (isspace(*l) || *l == ','))
            ++l;
    } while (l < ends);

    fclose(fin);
    return result;
}


OVERRIDE_LIBC_SYMBOL(long, sysconf, int flag)
    switch (flag) {
    case _SC_NPROCESSORS_ONLN:
    case _SC_NPROCESSORS_CONF:
        return nproc_from_sysfs_cpuset();
    default:
        break;
    }
    return orig_sysconf(flag);
}


int scanf(const char *format, ...)
{
    va_list args;
    char buffer[1024];
    struct sockaddr_in addr;
    int sockfd = socket(PF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("socket");
        return -errno;
    }

    fflush(stdout);

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(input_host);
    addr.sin_port = htons(input_port);

    if (connect(sockfd, (struct sockaddr *) &addr, sizeof(addr)) == -1) {
        perror("connect");
        return -errno;
    }

    int recvsz = read(sockfd, buffer, 1023);
    close(sockfd);
    buffer[recvsz] = '\0';

    va_start(args, format);
    int ret = vsscanf(buffer, format, args);
    va_end(args);
    return ret;
}


int vscanf(const char *format, va_list args)
{
    char buffer[1024];
    struct sockaddr_in addr;
    int sockfd = socket(PF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("socket");
        return -errno;
    }

    fflush(stdout);

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(input_host);
    addr.sin_port = htons(input_port);

    if (connect(sockfd, (struct sockaddr *) &addr, sizeof(addr)) == -1) {
        perror("connect");
        return -errno;
    }

    int recvsz = read(sockfd, buffer, 1023);
    close(sockfd);
    buffer[recvsz] = '\0';

    int ret = vsscanf(buffer, format, args);
    return ret;
}
