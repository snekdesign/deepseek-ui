import asyncio
import os
import socket
import string
import subprocess
import sys
from collections.abc import Sequence

import ollama
import pooch  # pyright: ignore[reportMissingTypeStubs]
import tqdm

from . import _deepseek_ui


def main():
    with socket.socket() as sock:
        try:
            sock.bind(_default_env())
        except OSError:
            pass
        else:
            _main(sock)


def _default_env():
    os.environ.setdefault('OLLAMA_CONTEXT_LENGTH', '90000')

    if host := os.getenv('OLLAMA_HOST'):
        host, port = host.split(':')
        port = int(port)
    else:
        [host] = _deepseek_ui.ollama_host()
        port = 8000
        os.environ['OLLAMA_HOST'] = f'{host}:8000'

    for drive in string.ascii_uppercase:
        if os.path.isdevdrive(f'{drive}:\\'):
            os.environ['OLLAMA_MODELS'] = fr'{drive}:\.cache\ollama'
            break
    return host, port


def _executable(files: Sequence[str]):
    for fullname in files:
        if os.path.basename(fullname) == 'ollama.exe':
            return fullname
    assert False, 'unreachable'


async def _load_model(ollama_exe: str) -> int:
    # https://github.com/ollama/ollama/blob/main/docs/api.md#load-a-model-1
    client = ollama.AsyncClient()
    while True:
        try:
            response = await client.chat('maternion/mimo-v2.6:9b', messages=[])  # pyright: ignore[reportUnknownMemberType]
            break
        except ollama.ResponseError as e:
            if e.status_code != 404:
                raise
            with tqdm.tqdm(unit='B', unit_scale=True) as prog:
                async for r in await client.pull('maternion/mimo-v2.6:9b', stream=True):
                    prog.set_description(r.status)
                    if total := r.total:
                        prog.total = total
                    if n := r.completed:
                        prog.update(n - prog.n)
        except Exception:
            await asyncio.sleep(1)
    if not (
        response.done and response.done_reason == 'load'
        and response.model == 'maternion/mimo-v2.6:9b'
    ):
        raise RuntimeError(response)
    process = await asyncio.create_subprocess_exec(
        ollama_exe, 'ps',
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
    )
    stdout, _ = await process.communicate()
    if returncode := await process.wait():
        raise subprocess.CalledProcessError(
            returncode=returncode,
            cmd=[ollama_exe, 'ps'],
        )
    if b'CPU' in stdout:  # This information is unavailable in API
        raise RuntimeError(stdout.decode())
    sys.stdout.buffer.write(stdout)
    sys.stdout.flush()
    os._exit(0)


def _main(sock: socket.socket):
    files = pooch.retrieve(  # pyright: ignore[reportUnknownMemberType]
        'https://mirror.nyist.edu.cn/github-release/ollama/ollama/LatestRelease/ollama-windows-amd64.zip',
        known_hash=_sha256(),
        processor=pooch.Unzip([
            'ollama.exe',
            'lib/ollama/cuda_v13/ggml-cuda.dll',
            'lib/ollama/ggml-base.dll',
            'lib/ollama/ggml-cpu-haswell.dll',
            'lib/ollama/ggml.dll',
            'lib/ollama/libc++.dll',
            'lib/ollama/libllama-common.dll',
            'lib/ollama/libllama-server-impl.dll',
            'lib/ollama/libllama.dll',
            'lib/ollama/libmtmd.dll',
            'lib/ollama/libunwind.dll',
            'lib/ollama/llama-server.exe',
        ]),
        downloader=_downloader,  # pyright: ignore[reportArgumentType]
    )
    ollama = _executable(files)
    sock.close()
    asyncio.run(_server(ollama))


async def _server(ollama: str):
    process = await asyncio.create_subprocess_exec(
        ollama, 'serve',
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    proc = asyncio.create_task(process.wait())
    done, pending = await asyncio.wait(
        [proc, asyncio.create_task(_load_model(ollama))],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    if proc in done and proc.exception() is None:
        raise subprocess.CalledProcessError(
            returncode=proc.result(),
            cmd=[ollama, 'serve'],
        )
    assert False, 'unreachable'


def _sha256():
    sha256sum = pooch.retrieve(  # pyright: ignore[reportUnknownMemberType]
        'https://mirror.nyist.edu.cn/github-release/ollama/ollama/LatestRelease/sha256sum.txt',
        downloader=_downloader,  # pyright: ignore[reportArgumentType]
    )
    with open(sha256sum, encoding='ascii') as f:
        for line in f:
            if 'ollama-windows-amd64.zip' in line:
                return line[:64]
    assert False, 'unreachable'


_downloader = pooch.HTTPDownloader(headers={'User-Agent': 'curl/8.22.0'})
