// Original CLI version written by DeepSeek-V4.1-Flash
// Reviewed by Qwen3.8-Max
// Converted to nanobind by hand

#include <cstdlib>
#include <new>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/wstring.h>
#include <Python.h>

#ifndef UNICODE
#  define UNICODE
#endif
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include <Windows.h>

namespace {
auto OllamaHost() {
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) [[unlikely]] {
        nanobind::raise("WSAStartup failed");
    }

    if (
        LOBYTE(wsaData.wVersion) != 2
        || HIBYTE(wsaData.wVersion) != 2
    ) [[unlikely]] {
        WSACleanup();
        nanobind::raise("Requested Winsock version not supported");
    }

    ULONG flags = GAA_FLAG_SKIP_ANYCAST |
                  GAA_FLAG_SKIP_MULTICAST |
                  GAA_FLAG_SKIP_DNS_SERVER;

    PIP_ADAPTER_ADDRESSES addresses {};
    ULONG outBufLen {};
    DWORD ret {};
    std::vector<std::basic_string<TCHAR>> found;

    for (int attempt = 0; attempt < 5; ++attempt) {
        if (outBufLen != 0) {
            auto tmp = std::realloc(addresses, outBufLen);
            if (tmp == nullptr) [[unlikely]] {
                std::free(addresses);
                WSACleanup();
                throw std::bad_alloc();
            }
            addresses = reinterpret_cast<PIP_ADAPTER_ADDRESSES>(tmp);
        }

        ret = GetAdaptersAddresses(
            AF_INET,
            flags,
            nullptr,
            addresses,
            &outBufLen);

        if (ret != ERROR_BUFFER_OVERFLOW) {
            break;
        }

        if (outBufLen == 0) [[unlikely]] {
            std::free(addresses);
            WSACleanup();
            return found;
        }
    }

    if (ret != NO_ERROR) [[unlikely]] {
        std::free(addresses);
        WSACleanup();
        if (auto args = Py_BuildValue(
            "(sssk)",
            /*errno=*/nullptr,
            /*strerror=*/"GetAdaptersAddresses failed",
            /*filename=*/nullptr,
            /*winerror=*/ret
        )) [[likely]] {
            PyErr_SetObject(PyExc_OSError, args);
        }
        throw nanobind::python_error();
    }

    if (addresses == nullptr) [[unlikely]] {
        WSACleanup();
        return found;
    }

    for (auto p = addresses; p != nullptr; p = p->Next) {
        if (p->OperStatus != IfOperStatusUp) continue;
        if (p->IfType == IF_TYPE_SOFTWARE_LOOPBACK) continue;

        for (auto u = p->FirstUnicastAddress; u != nullptr; u = u->Next) {
            if (u->Address.lpSockaddr == nullptr) continue;
            if (u->Address.iSockaddrLength < sizeof(SOCKADDR_IN)) continue;
            if (u->Address.lpSockaddr->sa_family != AF_INET) continue;
            if (u->PrefixOrigin != IpPrefixOriginDhcp) continue;

            auto sin = (SOCKADDR_IN *)u->Address.lpSockaddr;
            TCHAR ip[INET_ADDRSTRLEN];

            if (!InetNtop(AF_INET, &sin->sin_addr, ip, 16)) [[unlikely]] {
                std::free(addresses);
                if (auto args = Py_BuildValue(
                    "(sssi)",
                    /*errno=*/nullptr,
                    /*strerror=*/"InetNtop failed",
                    /*filename=*/nullptr,
                    /*winerror=*/WSAGetLastError()
                )) [[likely]] {
                    PyErr_SetObject(PyExc_OSError, args);
                }
                WSACleanup();
                throw nanobind::python_error();
            }

            found.emplace_back(ip);
        }
    }

    std::free(addresses);
    WSACleanup();
    return found;
}
}  // namespace

NB_MODULE(_deepseek_ui, m) {
    m.def("ollama_host", OllamaHost);
}
