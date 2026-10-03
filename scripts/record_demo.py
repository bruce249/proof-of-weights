"""Record the local slash dashboard. Requires the dashboard on port 8765."""

from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "demo"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            record_video_dir=str(OUT),
            record_video_size={"width": 1280, "height": 800},
        )
        page = context.new_page()
        page.goto("http://127.0.0.1:8765", wait_until="networkidle")
        page.get_by_role("heading", name="CAUGHT").wait_for()
        page.wait_for_timeout(2500)
        page.mouse.wheel(0, 500)
        page.wait_for_timeout(1500)
        video = page.video
        context.close()
        browser.close()
        if video is not None:
            src = Path(video.path())
            dest = OUT / "slash-demo.webm"
            if dest.exists():
                dest.unlink()
            src.replace(dest)
            print(dest)


if __name__ == "__main__":
    main()
