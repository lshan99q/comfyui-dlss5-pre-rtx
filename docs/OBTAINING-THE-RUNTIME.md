# Obtaining `nvngx_dlssnr.dll`

This project needs one file it cannot ship: **`nvngx_dlssnr.dll`**, the DLSS 5 Neural Rendering
runtime. It is proprietary NVIDIA software, it is not licensed for redistribution, and this
repository does not link to mirrors of it. Here is where it legitimately comes from and how to
tell whether the copy you have is usable.

## Short answer

It ships inside **games that carry DLSS 5**. The first and currently best-documented source is
**NBA 2K27**. Copy it out of a copy of the game you own.

## It is not in NVIDIA's public SDKs

Both obvious first-party sources were checked and neither contains it:

| Source | Checked | Contains `nvngx_dlssnr.dll`? |
| --- | --- | --- |
| [Streamline SDK v2.14.1](https://github.com/NVIDIA-RTX/Streamline/releases) (2026-09-08) | `bin/x64/` inspected | ❌ only `nvngx_dlss`, `nvngx_dlssd`, `nvngx_dlssg`, `nvngx_deepdvc` |
| [DLSS SDK v310.9.1 demo](https://github.com/NVIDIA/DLSS/releases) (2026-09-08) | zip central directory parsed, 1826 entries | ❌ only `DLSS_Sample_App/bin/ngx_dlss_demo/nvngx_dlss.dll` |

This holds even though DLSS 5 launched on 2026-09-03. NVIDIA distributes the neural-rendering
runtime to game developers, not through the public SDK — so a game is the only route.

## The known source: NBA 2K27

The file was first spotted in **NBA 2K27**'s early-access build on 2026-08-27 (found by Renan
Maniero; covered by [Guru3D](https://www.guru3d.com/story/first-dlss-5-runtime-discovered-in-nba-2k27-pc-build/),
[IT之家](https://m.ithome.com/html/995197.htm), [TweakTown](https://www.tweaktown.com/news/113305/nvidias-dlss-5-neural-rendering-tech-has-been-found-in-the-first-publicly-accessible-game/index.html)).
DLSS 5 shipped publicly with the game on 2026-09-03.

Reported properties of that file:

| Property | Value |
| --- | --- |
| Name | `nvngx_dlssnr.dll` |
| File version | `310.8.0.0` |
| Size | 165,840,496 bytes (158 MiB) |
| PE modification date | 2026-08-26 |

For scale, ordinary DLSS Super Resolution DLLs are 20–50 MB. This one is roughly three times
the largest DLSS 4 file, which is what a 71-block transformer costs.

### Finding it in a game install

Search the game's installation directory — the DLL typically sits next to the shipping
executable or in a plugin subfolder:

```bash
# Linux / Steam Proton
find ~/.steam/steam/steamapps/common -iname "nvngx_dlssnr.dll" 2>/dev/null
find ~/.local/share/Steam/steamapps/common -iname "nvngx_dlssnr.dll" 2>/dev/null

# everything, if you have multiple drives mounted
find / -iname "nvngx_dlssnr.dll" 2>/dev/null | grep -v "^/proc"
```

Windows:

```powershell
Get-ChildItem -Path C:\ -Filter nvngx_dlssnr.dll -Recurse -ErrorAction SilentlyContinue |
    Select-Object FullName, Length, LastWriteTime
```

### Verify the copy before using it

Do not trust the version string alone — the extractor accepts one canonical SHA-256, and other
`310.8.0.0` builds exist.

```bash
# what is this file?
python -m mlxdlss.tools.cli sha256 /path/to/nvngx_dlssnr.dll
#   -> <sha256>  version 310,8,0,0  <known checkpoint | unknown checkpoint>

# does it contain a network we can decode?
python -m mlxdlss.tools.cli extract /path/to/nvngx_dlssnr.dll /tmp/packed.safetensors
python -m mlxdlss.tools.cli decode  /tmp/packed.safetensors /tmp/logical.safetensors
```

A usable build decodes to **153 source tensors → 649 logical tensors, `unsupportedSourceTensorCount: 0`,
`opaqueOutputTensorCount: 0`, format `dlssnr-logical-v18`**. `verify_pre_rtx.py` checks exactly this.

> [!NOTE]
> The extractor prints `unknown checkpoint` for builds whose SHA-256 it does not recognise. That
> is **not** automatically a failure: a different `310.8.0.0` build was verified here to decode
> to the identical 153 → 649 structure and to produce correct output. The structural check is
> the one that matters.

## What will not work

- **Renaming another DLL.** `nvngx_dlss.dll` (Super Resolution), `nvngx_dlssg.dll` (frame
  generation) and `nvngx_dlssd.dll` (ray reconstruction) are different components with
  different weights. Renaming one does not produce a neural-rendering runtime.
- **An older or newer build.** The decoder targets the `WEIGHTS_HT` layout of `310.8.0.0`.
- **DLLs from driver packages.** Drivers ship loaders (`_nvngx.dll`, `nvngx.dll`), not this.

## The feature being present in a game ≠ enabled

The DLL sitting in NBA 2K27's files does not mean the game renders with DLSS 5. Neural rendering
needs engine-side integration — the game must feed geometry, material and lighting buffers. This
project bypasses that entirely by running the recovered network directly on images, which is why
it works at all without a supporting game.

## Licensing

`nvngx_dlssnr.dll` is NVIDIA property, licensed for use with the product it ships in. Copying it
out of a game you own for personal use is what the community does; **redistributing it is not
permitted**, which is why this repository, MLX-DLSS, and every sibling project all require you
to bring your own copy.

If you do not own a DLSS 5 game, wait for NVIDIA to publish the runtime — given DLSS 5 is
generally available as of 2026-09-03, it may reach the public SDK eventually.
