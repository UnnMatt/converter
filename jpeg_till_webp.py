"""

ANVÄNDNING:
    python jpeg_till_webp.py <in-mapp> <ut-mapp> [flaggor]

EXEMPEL:
    python jpeg_till_webp.py ./in ./ut --quality 80 --method 4 --workers 24
    python jpeg_till_webp.py ./in ./ut --max-side 2560 --method 5
    python jpeg_till_webp.py ./in ./ut --dry-run
"""

from __future__ import annotations
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# ----------------------------
# Grundinställningar – ÄNDRA HÄR vid behov
# ----------------------------
DEFAULT_QUALITY = 80           # 75–85 ger bra balans foto/kvalitet
DEFAULT_METHOD = 4             # 0–6 (högre = mindre fil men långsammare)
DEFAULT_WORKERS = os.cpu_count() or 8  # i7-14700K: testa 20–28
DEFAULT_SAMPLE_BENCH = 80      # hur många filer som testas för ETA
DEFAULT_MAX_SIDE = None        # t.ex. 2560 för att skala ner längsta sidan
PRESERVE_MTIME = True          # behåll källfilens ändringstid
OVERWRITE = False              # True = skriv över redan konverterade
DRY_RUN = False                # True = kör utan att spara filer
KEEP_METADATA = True           # <--- ÄNDRA TILL False för att STRIPPA metadata (EXIF/ICC)

# ----------------------------
# Hjälpfunktioner
# ----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG"}

@dataclass
class EncodeOptions:
    quality: int
    method: int
    lossless: bool
    near_lossless: Optional[int]
    max_side: Optional[int]
    preserve_metadata: bool
    preserve_mtime: bool
    overwrite: bool
    dry_run: bool

def iter_jpegs(root: Path) -> List[Path]:
    """Hittar alla JPEG-filer rekursivt i en mapp."""
    return [p for p in root.rglob("*") if p.suffix in IMG_EXTS and p.is_file()]

def dest_path(src: Path, src_root: Path, dst_root: Path) -> Path:
    """Bygger målfilens sökväg (behåller mappstruktur) och byter till .webp."""
    rel = src.relative_to(src_root)
    return (dst_root / rel).with_suffix(".webp")

def resize_if_needed(im: Image.Image, max_side: Optional[int]) -> Image.Image:
    """Skalar ner bilden om längsta sidan överstiger max_side."""
    if not max_side:
        return im
    w, h = im.size
    longest = max(w, h)
    if longest <= max_side:
        return im
    scale = max_side / float(longest)
    new_size = (int(w * scale), int(h * scale))
    # LANCZOS ger bra kvalitet vid nedskalning (snabbare med pillow-simd)
    return im.resize(new_size, Image.LANCZOS)

def encode_one(src: Path, src_root: Path, dst_root: Path,
               opts: EncodeOptions) -> Tuple[Path, float, Optional[str]]:
    """Konverterar en bild och returnerar (källa, tid i sek, ev. felmeddelande)."""
    t0 = time.time()
    outp = dest_path(src, src_root, dst_root)

    try:
        outp.parent.mkdir(parents=True, exist_ok=True)

        # hoppa över om utfil redan finns och vi inte ska skriva över
        if outp.exists() and not opts.overwrite:
            return (src, 0.0, None)
        if opts.dry_run:
            return (src, 0.0, None)

        with Image.open(src) as im:
            # säkerställ lämpligt färgläge
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGBA" if "A" in im.getbands() else "RGB")

            # ev. nedskalning före komprimering
            im = resize_if_needed(im, opts.max_side)

            # bygg spar-parametrar
            save_kwargs = {}

            # --- kvalitet/metod/lägen ---
            if opts.lossless:
                # helt förlustfritt (större filer, långsammare)
                save_kwargs.update({"lossless": True, "method": opts.method})
            elif opts.near_lossless is not None:
                # near-lossless (bra för grafik/UI/skarpa kanter)
                save_kwargs.update({
                    "quality": opts.quality,
                    "method": opts.method,
                    "near_lossless": opts.near_lossless,
                })
            else:
                # standard (lossy) för foton
                save_kwargs.update({"quality": opts.quality, "method": opts.method})

            # --- metadata (styrt av KEEP_METADATA) ---
            if opts.preserve_metadata:
                if "exif" in im.info:
                    save_kwargs["exif"] = im.info["exif"]
                if "icc_profile" in im.info:
                    save_kwargs["icc_profile"] = im.info["icc_profile"]
            # Om KEEP_METADATA=False skickas inget EXIF/ICC → metadata strippas

            im.save(outp, "WEBP", **save_kwargs)

        # behåll ändringstid om valt
        if opts.preserve_mtime:
            try:
                stat = src.stat()
                os.utime(outp, (stat.st_atime, stat.st_mtime))
            except Exception:
                pass

        return (src, time.time() - t0, None)

    except (UnidentifiedImageError, OSError) as e:
        return (src, 0.0, f"hoppar över: {e}")
    except Exception as e:
        return (src, 0.0, f"fel: {e}")

def benchmark(files: List[Path], src_root: Path, dst_root: Path,
              opts: EncodeOptions, sample_n: int) -> Tuple[int, float]:
    """Testar ett mindre antal filer för att uppskatta tid per fil."""
    sample = files[: min(sample_n, len(files))]
    done, secs = 0, 0.0
    for p in sample:
        _, dt, err = encode_one(p, src_root, dst_root, opts)
        if err is None:
            secs += max(dt, 1e-6)
            done += 1
    return done, secs

def main():
    ap = argparse.ArgumentParser(description="Konvertera JPEG → WebP (parallellt, behåll/strippa metadata via KEEP_METADATA)")
    ap.add_argument("src", type=Path, help="Mapp med JPEG-bilder (skannas rekursivt)")
    ap.add_argument("dst", type=Path, help="Mapp där WebP-filer sparas (mappstruktur behålls)")
    ap.add_argument("--quality", type=int, default=DEFAULT_QUALITY, help="Kvalitet 0–100 (lossy-läge)")
    ap.add_argument("--method", type=int, default=DEFAULT_METHOD, help="Komprimeringsmetod 0–6 (högre = mindre fil men långsammare)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Antal parallella processer")
    ap.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE, help="Skala längsta sidan till detta antal pixlar (valfritt)")
    ap.add_argument("--lossless", action="store_true", help="Använd helt förlustfri WebP")
    ap.add_argument("--near-lossless", type=int, default=None, metavar="N", help="Near-lossless 0–100 (lägre ≈ mer förlustfritt)")
    ap.add_argument("--no-preserve-mtime", action="store_true", help="Behåll inte ändringstid från källfil")
    ap.add_argument("--overwrite", action="store_true", help="Skriv över om .webp redan finns")
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE_BENCH, help="Antal filer för benchmark (ETA)")
    ap.add_argument("--dry-run", action="store_true", help="Beräkna ETA men spara inga filer")
    args = ap.parse_args()

    src_root: Path = args.src
    dst_root: Path = args.dst
    dst_root.mkdir(parents=True, exist_ok=True)

    opts = EncodeOptions(
        quality=args.quality,
        method=max(0, min(6, args.method)),
        lossless=args.lossless,
        near_lossless=args.near_lossless,
        max_side=args.max_side,
        preserve_metadata=KEEP_METADATA,            # <-- styrs av globala flaggan
        preserve_mtime=not args.no_preserve_mtime,
        overwrite=args.overwrite,
        dry_run=args.dry_run
    )

    files = iter_jpegs(src_root)
    if not files:
        print("Inga JPEG-filer hittades.")
        return

    print(f"Hittade {len(files)} JPEG-bilder i {src_root}")
    print(f"Utdatamapp: {dst_root}")
    mode = "lossless" if opts.lossless else ("near-lossless" if opts.near_lossless is not None else "lossy")
    print(f"Läge: {mode} | kvalitet={opts.quality} metod={opts.method} workers={args.workers} max_side={opts.max_side or 'None'}")
    print(f"Metadata: {'BEHÅLLS' if opts.preserve_metadata else 'STRIPPAS'} (KEEP_METADATA={KEEP_METADATA})")
    if opts.dry_run:
        print("Torrkörning – inga filer sparas.")

    # --- Benchmark för ETA ---
    print(f"Benchmark på första {min(args.sample, len(files))} filer...")
    done, secs = benchmark(files, src_root, dst_root, opts, args.sample)
    if done > 0 and secs > 0:
        per_file = secs / done
        # enkel uppskattning: (arbetare / tid per fil)
        est_rate = args.workers / per_file
        remaining = len(files) - done
        eta_sec = remaining / est_rate if est_rate > 0 else 0
        print(f"{done} filer på {secs:.1f}s → ~{(done/secs):.2f} bilder/s (en process).")
        print(f"Beräknad tid med {args.workers} arbetare: ~{eta_sec/60:.1f} min för {remaining} kvar.")
    else:
        print("Benchmark gav inget resultat.")

    if opts.dry_run:
        print("Klart (torrkörning).")
        return

    # --- Parallell konvertering ---
    errors = 0
    start_all = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(encode_one, p, src_root, dst_root, opts) for p in files]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Konverterar", unit="bild"):
            _, _, err = fut.result()
            if err:
                errors += 1

    dur = time.time() - start_all
    print(f"Klart på {dur/60:.1f} minuter. Utdata: {dst_root}")
    if errors:
        print(f"Skippade {errors} filer p.g.a. fel.")
    else:
        print("Alla filer konverterade utan fel.")

if __name__ == "__main__":
    main()
