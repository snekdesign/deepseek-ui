import asyncio
import os
import socket
import subprocess
from typing import List

import ollama
import pooch  # pyright: ignore[reportMissingTypeStubs]
import torch.version


def main():
    with socket.socket() as sock:
        try:
            sock.bind(('127.0.0.1', 11434))
        except OSError:
            pass
        else:
            _main(sock)


def _cuda_version():
    if version := torch.version.cuda:
        if version.startswith('12.'):
            return 12
        if version.startswith('11.'):
            return 11
    assert False, 'unreachable'


def _executable(files: List[str]):
    for fullname in files:
        if os.path.basename(fullname) == 'ollama.exe':
            return fullname
    assert False, 'unreachable'


async def _load_model():
    # https://github.com/ollama/ollama/blob/main/docs/api.md#load-a-model-1
    client = ollama.AsyncClient()
    while True:
        try:
            response = await client.chat('deepseek-r1:8b', messages=[])  # pyright: ignore[reportUnknownMemberType]
            break
        except Exception:
            await asyncio.sleep(1)
    if (
        response.done and response.done_reason == 'load'
        and response.model == 'deepseek-r1:8b'
    ):
        os._exit(0)
    raise RuntimeError(response)


def _main(sock: socket.socket):
    major = _cuda_version()
    files = pooch.retrieve(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        'https://mirror.nyist.edu.cn/github-release/ollama/ollama/LatestRelease/ollama-windows-amd64.zip',
        known_hash=_sha256(),
        processor=pooch.Unzip([
            'ollama.exe',
            f'lib/ollama/runners/cuda_v{major}_avx/ggml_cuda_v{major}.dll',
            f'lib/ollama/runners/cuda_v{major}_avx/ollama_llama_server.exe',
        ]),
    )
    assert isinstance(files, list)
    ollama = _executable(files)  # pyright: ignore[reportUnknownArgumentType]
    sock.close()
    asyncio.run(_server(ollama))


async def _server(ollama: str):
    process = await asyncio.create_subprocess_exec(
        ollama, 'serve',
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    proc = asyncio.create_task(process.wait())
    done, pending = await asyncio.wait(
        [proc, asyncio.create_task(_load_model())],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    if proc.done() and proc.exception() is None:
        raise subprocess.CalledProcessError(
            returncode=proc.result(),
            cmd=[ollama, 'serve'],
        )
    for task in done:
        await task
    assert False, 'unreachable'


def _sha256():
    sha256sum = pooch.retrieve(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        'https://mirror.nyist.edu.cn/github-release/ollama/ollama/LatestRelease/sha256sum.txt',
        known_hash=None,
    )
    assert isinstance(sha256sum, str)
    with open(sha256sum, encoding='ascii') as f:
        for line in f:
            if 'ollama-windows-amd64.zip' in line:
                return line[:64]
    assert False, 'unreachable'


if __name__ == '__main__':
    main()
