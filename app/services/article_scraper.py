import re
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Sequence
from urllib.parse import urlparse

from playwright.async_api import async_playwright, TimeoutError as PWTimeout

logger = logging.getLogger(__name__)

CONSENT_HOST_PATTERNS = (
    r"(^|\.)consent\.",
    r"(^|\.)cmp\.",
    r"(^|\.)consentmanager\.net",
    r"(^|\.)quantcast\.com",
    r"(^|\.)privacymanager\.io",
)

CONSENT_TEXT_HINTS = (
    "cookie", "cookies", "gdpr", "consent", "privacy",
    "agree", "accept", "manage preferences", "your choices",
)

CONSENT_CLICK_SELECTORS = [
    "button:has-text('Accept All')",
    "button:has-text('Accept all')",
    "button:has-text('I Agree')",
    "button:has-text('I accept')",
    "button[aria-label*='Accept']",
    "button[aria-label*='agree']",
    "[role='dialog'] button:has-text('Accept')",
    "[data-testid='uc-accept-all-button']",
    "#onetrust-accept-btn-handler",
    "button#onetrust-accept-btn-handler",
    ".fc-cta-consent .fc-button-label",
    "button[mode='primary']:has-text('Accept')",
]


@dataclass
class ScrapeResult:
    url: str
    title: str
    text: str
    selector: Optional[str]
    method: str
    skipped: bool
    skipped_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScraperConfig:
    output_json: str = "output/articles.json"
    raw_text_dir: str = "output/raw_text"
    headless: bool = False
    max_articles: int = 0
    selectors: Optional[Sequence[str]] = None
    viewport: Optional[Dict[str, int]] = None
    user_agent: Optional[str] = None


class ConsentManager:
    @staticmethod
    def host_matches_consent(host: str) -> bool:
        for pat in CONSENT_HOST_PATTERNS:
            if re.search(pat, host, flags=re.IGNORECASE):
                return True
        return False

    @staticmethod
    async def looks_like_consent_overlay(page) -> bool:
        try:
            containers = page.locator("div, section, aside, [role='dialog'], [role='alertdialog']")
            count = min(await containers.count(), 40)
            for i in range(count):
                handle = containers.nth(i)
                if await handle.is_visible():
                    text = (await handle.inner_text(timeout=500) or "").lower()
                    if any(h in text for h in CONSENT_TEXT_HINTS):
                        if await handle.locator("button").count() > 0:
                            return True
        except Exception:
            pass
        return False

    @staticmethod
    async def try_accept_consent(page) -> None:
        for sel in CONSENT_CLICK_SELECTORS:
            try:
                btn = page.locator(sel)
                if await btn.first.is_visible():
                    await btn.first.click(timeout=800)
                    await page.wait_for_timeout(400)
                    break
            except Exception:
                continue


class PlaywrightArticleScraper:

    def __init__(
        self,
        headless: bool = False,
        selectors: Optional[Sequence[str]] = None,
        viewport: Optional[Dict[str, int]] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        self.headless = headless
        self.selectors = list(selectors or [
            "article",
            "main article",
            "main",
            "section[role='main']",
            "div.article__main",
        ])
        if viewport is not None:
            self.width = int(viewport.get("width", 1366))
            self.height = int(viewport.get("height", 850))
        else:
            self.width = 1366
            self.height = 850
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        )

    async def scrape(self, url: str) -> ScrapeResult:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            try:
                context = await browser.new_context(
                    ignore_https_errors=True,
                    user_agent=self.user_agent,
                    viewport={"width": self.width, "height": self.height},
                    locale="en-US",
                    timezone_id="UTC",
                )
                page = await context.new_page()

                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=90000)
                except PWTimeout:
                    await page.goto(url, wait_until="load", timeout=120000)

                try:
                    parsed = urlparse(page.url)
                    if ConsentManager.host_matches_consent(parsed.hostname or ""):
                        return ScrapeResult(
                            url=url,
                            title="",
                            text="",
                            selector=None,
                            method="playwright-async",
                            skipped=True,
                            skipped_reason=f"Redirected to consent host: {parsed.hostname}",
                        )
                except Exception:
                    pass

                await ConsentManager.try_accept_consent(page)

                for _ in range(3):
                    await page.mouse.wheel(0, 1500)
                    await page.wait_for_timeout(400)

                title = (await page.title() or "").strip()
                text = ""
                used_selector = None

                for sel in self.selectors:
                    try:
                        await page.wait_for_selector(sel, timeout=6000)
                        used_selector = sel
                        paras = await page.locator(f"{sel} p").all_text_contents()
                        if paras:
                            text = "\n\n".join([t.strip() for t in paras if t and t.strip()])
                        else:
                            text = (await page.locator(sel).text_content() or "").strip()
                        if text and len(text.split()) > 30:
                            break
                    except Exception:
                        continue

                if not text or len(text.split()) < 30:
                    if await ConsentManager.looks_like_consent_overlay(page):
                        return ScrapeResult(
                            url=url,
                            title=title,
                            text="",
                            selector=used_selector,
                            method="playwright-async",
                            skipped=True,
                            skipped_reason="Consent/Cookie overlay blocking content",
                        )

                return ScrapeResult(
                    url=url,
                    title=title,
                    text=text,
                    selector=used_selector,
                    method="playwright-async",
                    skipped=False,
                )

            finally:
                try:
                    await context.close()
                except Exception:
                    pass
                await browser.close()