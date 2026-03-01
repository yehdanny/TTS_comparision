"""
Take screenshots of the TTS Comparison web page for the README.
Run from project root: python take_screenshots.py
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

PAGE_URL = (Path(__file__).parent / "frontend" / "index.html").resolve().as_uri()
DOCS_DIR = Path(__file__).parent / "docs"


async def main() -> None:
    DOCS_DIR.mkdir(exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1400, "height": 900})

        await page.goto(PAGE_URL, wait_until="networkidle")

        # ── Inject mock state so model cards look populated ───────────────────
        await page.evaluate("""() => {
            // Set all status badges to Offline (backend not running — that's fine)
            ['f5','cosyvoice','gptsovits'].forEach(m => {
                const el = document.getElementById('status-' + m);
                if (el) { el.className = 'status-badge status-offline'; el.textContent = 'Offline'; }
            });

            // Populate metrics with realistic stub values
            const data = {
                f5:        { time: '0.41 s', rtf: '0.410', size: '16.2 KB' },
                cosyvoice: { time: '0.53 s', rtf: '0.530', size: '16.2 KB' },
                gptsovits: { time: '0.67 s', rtf: '0.670', size: '22.1 KB' },
            };
            for (const [m, d] of Object.entries(data)) {
                document.getElementById('metric-' + m + '-time').textContent = d.time;
                document.getElementById('metric-' + m + '-rtf').textContent  = d.rtf;
                document.getElementById('metric-' + m + '-size').textContent = d.size;
                // Show a visible audio section placeholder
                const sec = document.getElementById('audio-section-' + m);
                if (sec) sec.style.display = 'flex';
            }
        }""")

        await page.wait_for_timeout(300)

        # ── Screenshot 1: header + input + model cards ────────────────────────
        main_section = await page.query_selector("main")
        # Clip to just the top portion (input panel + model grid)
        model_grid = await page.query_selector(".model-grid")
        grid_box   = await model_grid.bounding_box()
        await page.screenshot(
            path=str(DOCS_DIR / "screenshot_main.png"),
            clip={
                "x": 0, "y": 0,
                "width": 1400,
                "height": grid_box["y"] + grid_box["height"] + 40,
            },
        )
        print("Saved docs/screenshot_main.png")

        # ── Screenshot 2: demo samples section (full height) ─────────────────
        # Expand viewport to fit the entire page so nothing is clipped
        total_height = await page.evaluate("document.body.scrollHeight")
        await page.set_viewport_size({"width": 1400, "height": total_height})
        await page.wait_for_timeout(200)

        demo_section = await page.query_selector(".demo-section")
        demo_box     = await demo_section.bounding_box()
        await page.screenshot(
            path=str(DOCS_DIR / "screenshot_demo.png"),
            clip={
                "x": 0,
                "y": demo_box["y"] - 20,
                "width": 1400,
                "height": demo_box["height"] + 40,
            },
        )
        print("Saved docs/screenshot_demo.png")

        await browser.close()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
