# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Clear the executable-stack flag (PF_X) on an ELF shared object's PT_GNU_STACK
program header. Needed because torch_xla's _XLAC.so ships with an executable stack,
which gVisor-sandboxed TPU workers refuse to load ("cannot enable executable stack").

Pure stdlib so it runs before torch_xla is importable. Usage: python clear_execstack.py <path...>
"""

import struct
import sys

PT_GNU_STACK = 0x6474E551
PF_X = 0x1


def clear(path: str) -> bool:
    with open(path, "r+b") as f:
        data = bytearray(f.read())
        if data[:4] != b"\x7fELF":
            raise ValueError(f"{path}: not an ELF file")
        is64 = data[4] == 2
        endian = "<" if data[5] == 1 else ">"
        if not is64:
            raise ValueError(f"{path}: only 64-bit ELF supported")
        e_phoff = struct.unpack_from(endian + "Q", data, 0x20)[0]
        e_phentsize = struct.unpack_from(endian + "H", data, 0x36)[0]
        e_phnum = struct.unpack_from(endian + "H", data, 0x38)[0]
        patched = False
        for i in range(e_phnum):
            off = e_phoff + i * e_phentsize
            p_type = struct.unpack_from(endian + "I", data, off)[0]
            if p_type == PT_GNU_STACK:
                # Elf64_Phdr: p_type(4) p_flags(4) ... -> p_flags at off+4
                flags = struct.unpack_from(endian + "I", data, off + 4)[0]
                if flags & PF_X:
                    struct.pack_into(endian + "I", data, off + 4, flags & ~PF_X)
                    patched = True
        if patched:
            f.seek(0)
            f.write(data)
    return patched


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"{p}: {'cleared PF_X' if clear(p) else 'no executable GNU_STACK'}")
