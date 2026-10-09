#!/usr/bin/env python3
"""Headless screenshots of the viewer, for checking it without a browser window.

    pip install playwright && python -m playwright install chromium
    # missing libnss3/libnspr4 and no sudo?  conda install -c conda-forge nss nspr
    python -m http.server 8765 -d bones/viewer &
    python bones/viewer/screenshot.py --out /tmp/viewer   # desktop light/dark + phone

Prints page errors and horizontal overflow for each shot.
"""
import argparse
import asyncio
import os
from pathlib import Path

from playwright.async_api import async_playwright

SHOTS = [("desktop_light", 1440, 900, "light"), ("desktop_dark", 1440, 900, "dark"), ("phone", 400, 860, "light")]


async def main(url: str, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if os.environ.get("CONDA_PREFIX"):  # nss/nspr from conda-forge, if installed there
        env["LD_LIBRARY_PATH"] = os.environ["CONDA_PREFIX"] + "/lib"
    async with async_playwright() as p:
        browser = await p.chromium.launch(env=env, args=["--use-gl=angle", "--use-angle=swiftshader",
                                                         "--enable-unsafe-swiftshader"])
        for name, w, h, scheme in SHOTS:
            page = await browser.new_page(viewport={"width": w, "height": h}, color_scheme=scheme)
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            await page.goto(url)
            await page.wait_for_timeout(4000)  # data fetch + WebGL first frames
            await page.screenshot(path=out / f"{name}.png", full_page=True)
            overflow = await page.evaluate("document.documentElement.scrollWidth") - w
            print(f"{name}: {'OK' if not errors and overflow <= 0 else 'CHECK'}"
                  f"  overflow={max(overflow, 0)}px  errors={errors or 'none'}")
        await browser.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8765/")
    ap.add_argument("--out", type=Path, default=Path("bones/work/screenshots"))
    a = ap.parse_args()
    asyncio.run(main(a.url, a.out))
