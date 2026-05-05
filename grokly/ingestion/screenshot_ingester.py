"""
grokly/ingestion/screenshot_ingester.py — Vision-based screenshot ingestion.

Sends each image to Claude Vision API, which generates a detailed text
description. Only the description is stored in ChromaDB — the original
image never leaves the client machine.

Supports: PNG, JPG, JPEG, GIF, WEBP, BMP
"""

from __future__ import annotations

import base64
import hashlib
import logging
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from grokly.model_config import get_agent_config
from grokly.store.chroma_store import ChromaStore

load_dotenv()

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS: dict[str, str] = {
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif":  "image/gif",
    ".webp": "image/webp",
    ".bmp":  "image/png",   # BMP converted to PNG before sending
}

_API_DELAY = 1.0  # seconds between Vision API calls


class ScreenshotIngester:
    """Describe screenshots with Claude Vision and store text in ChromaDB."""

    def __init__(self, store: ChromaStore) -> None:
        self._store  = store
        self._client = anthropic.Anthropic()
        self._cfg    = get_agent_config("screenshot")

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    def _encode_image(self, image_path: str) -> tuple[str, str]:
        """Return (base64_data, media_type) for the given image file."""
        path = Path(image_path)
        ext  = path.suffix.lower()

        if ext == ".bmp":
            from PIL import Image
            import io
            img    = Image.open(image_path)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            raw        = buffer.getvalue()
            media_type = "image/png"
        else:
            with open(image_path, "rb") as fh:
                raw = fh.read()
            media_type = SUPPORTED_FORMATS.get(ext, "image/jpeg")

        return base64.standard_b64encode(raw).decode("utf-8"), media_type

    # ------------------------------------------------------------------
    # Vision description
    # ------------------------------------------------------------------

    _SYSTEM = """\
You are an enterprise software documentation specialist analysing screenshots
of business applications.

Your descriptions will be stored in a knowledge base and searched by:
- End users learning how to use the system
- Business analysts understanding processes
- Developers understanding the UI
- UAT testers verifying functionality

Write descriptions that are:
- Specific about what is visible on screen
- Clear about what business process is shown
- Useful for someone trying to replicate the steps
- Searchable — include key terms someone would use

Do not begin with "The screenshot shows" — describe what is there directly.\
"""

    def _describe_screenshot(
        self,
        image_path: str,
        context:    str = "",
    ) -> str:
        """Send image to Claude Vision and return a structured description."""
        image_data, media_type = self._encode_image(image_path)
        filename = Path(image_path).stem

        context_line = f"Context: {context}\n" if context else ""

        response = self._client.messages.create(
            model=self._cfg["model"],
            max_tokens=self._cfg["max_tokens"],
            temperature=self._cfg["temperature"],
            system=self._SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": media_type,
                            "data":       image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Analyse this screenshot from a business application.\n\n"
                            f"Filename: {filename}\n"
                            f"{context_line}\n"
                            "Provide a comprehensive description covering:\n\n"
                            "1. SCREEN IDENTIFICATION\n"
                            "   What application screen or page is this?\n"
                            "   What is the screen title or transaction name?\n\n"
                            "2. BUSINESS PROCESS\n"
                            "   What business process or workflow step does this screen represent?\n"
                            "   Where does this fit in the overall process?\n\n"
                            "3. VISIBLE CONTENT\n"
                            "   What key fields, data, buttons, or options are visible?\n"
                            "   What values or entries are shown?\n\n"
                            "4. USER ACTIONS\n"
                            "   What actions can a user take from this screen?\n"
                            "   What is the next step after this screen?\n\n"
                            "5. ERRORS OR WARNINGS\n"
                            "   Are any error messages, warnings, or validation messages visible?\n"
                            "   What do they mean in business terms?\n\n"
                            "6. BUSINESS RULES\n"
                            "   What business rules or constraints are evident from this screen?\n\n"
                            "Keep the description under 300 words. "
                            "Use plain English that a business user would understand."
                        ),
                    },
                ],
            }],
        )
        return response.content[0].text.strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_single(
        self,
        image_path:  str,
        application: str = "unknown",
        context:     str = "",
    ) -> dict:
        """Describe and store one screenshot. Returns description and chunk_id."""
        description = self._describe_screenshot(image_path, context)
        path        = Path(image_path)
        chunk_id    = "screenshot-" + hashlib.md5(
            str(path.resolve()).encode()
        ).hexdigest()

        chunk_text = (
            f"Screenshot: {path.name}\n"
            f"Application: {application}\n\n"
            f"{description}"
        )

        self._store.upsert(
            texts=[chunk_text],
            metadatas=[{
                "source":         "screenshot",
                "chunk_type":     "screenshot",
                "file_path":      str(path),
                "file_name":      path.name,
                "application":    application,
                "ingestion_type": "vision",
            }],
            ids=[chunk_id],
        )

        return {
            "status":      "complete",
            "description": description,
            "chunk_id":    chunk_id,
        }

    def ingest_folder(
        self,
        folder_path:  str,
        application:  str  = "unknown",
        context_file: str  = "",
        dry_run:      bool = False,
        max_images:   int  = 9999,
    ) -> dict:
        """
        Ingest all images in *folder_path*.

        Parameters
        ----------
        context_file : optional path to a .txt file describing the screenshots;
                       first 500 chars are sent to Claude as context.
        """
        folder = Path(folder_path)
        if not folder.exists():
            return {"status": "error", "reason": f"Folder not found: {folder_path}"}

        # Load context hint
        context = ""
        if context_file:
            ctx_path = Path(context_file)
            if ctx_path.exists():
                context = ctx_path.read_text(encoding="utf-8", errors="ignore")[:500]

        # Discover images
        images: list[Path] = []
        for ext in SUPPORTED_FORMATS:
            images.extend(folder.glob(f"*{ext}"))
            images.extend(folder.glob(f"*{ext.upper()}"))
        images = sorted(set(images))[:max_images]

        if not images:
            return {"status": "no_images", "reason": f"No images found in {folder_path}"}

        print(f"\n[Screenshot Ingester]")
        print(f"  Folder:       {folder_path}")
        print(f"  Images found: {len(images)}")

        if dry_run:
            print("  DRY RUN — no processing")
            for img in images[:5]:
                print(f"    • {img.name}")
            if len(images) > 5:
                print(f"    ... and {len(images) - 5} more")
            return {"status": "dry_run", "image_count": len(images)}

        # Cost estimate: ~1000 input tokens/image (vision) + ~400 output
        # Sonnet 4.6 pricing: input $3/M, output $15/M
        est_cost = len(images) * ((1000 * 3.0 + 400 * 15.0) / 1_000_000)
        print(f"  Estimated cost: ${est_cost:.2f}")
        print("  Processing...\n")

        chunks_added = 0
        errors       = 0

        for i, image_path in enumerate(images, 1):
            print(f"  [{i}/{len(images)}] {image_path.name}", end="", flush=True)

            # Skip if already ingested
            chunk_id = "screenshot-" + hashlib.md5(
                str(image_path.resolve()).encode()
            ).hexdigest()
            existing = self._store._collection.get(ids=[chunk_id], include=[])
            if existing.get("ids"):
                print(" (skipped — already indexed)")
                continue

            try:
                description = self._describe_screenshot(str(image_path), context)
                chunk_text  = (
                    f"Screenshot: {image_path.name}\n"
                    f"Application: {application}\n"
                    f"Folder: {folder.name}\n\n"
                    f"{description}"
                )
                self._store.upsert(
                    texts=[chunk_text],
                    metadatas=[{
                        "source":         "screenshot",
                        "chunk_type":     "screenshot",
                        "file_path":      str(image_path),
                        "file_name":      image_path.name,
                        "folder":         folder.name,
                        "application":    application,
                        "ingestion_type": "vision",
                    }],
                    ids=[chunk_id],
                )
                chunks_added += 1
                print(" ✓")
                time.sleep(_API_DELAY)

            except Exception as exc:
                errors += 1
                print(f" ERROR: {exc}")
                logger.warning("Screenshot ingestion failed for %s: %s", image_path.name, exc)

        print(f"\n  Complete: {chunks_added} chunks added, {errors} errors")
        return {
            "status":           "complete",
            "chunks_added":     chunks_added,
            "errors":           errors,
            "images_processed": len(images),
        }
