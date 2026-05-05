"""
grokly/ingestion/recording_ingester.py — Vision-based screen recording ingestion.

Processing pipeline:
  1. Extract keyframes (1 per N seconds) using OpenCV
  2. Scene-change detection skips near-duplicate frames
  3. Claude Vision describes each keyframe (2-3 sentences)
  4. Optional Whisper audio transcription
  5. Frames grouped into ~30-second segments → one ChromaDB chunk each

Only text descriptions are stored — video frames never leave the client machine.

Supports: MP4, AVI, MOV, MKV, WEBM
Requires: pip install opencv-python
Optional: pip install openai-whisper  (for audio transcription)
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

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SEGMENT_DURATION = 30   # seconds per ChromaDB chunk
_API_DELAY       = 0.5  # seconds between Vision API calls


def _require_cv2():
    """Import cv2 or raise a helpful ImportError."""
    try:
        import cv2  # noqa: F401
        return cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for screen recording ingestion.\n"
            "Install with:  pip install opencv-python"
        )


class RecordingIngester:
    """Extract keyframes from screen recordings, describe them, and store in ChromaDB."""

    def __init__(self, store: ChromaStore) -> None:
        self._store  = store
        self._client = anthropic.Anthropic()
        self._cfg    = get_agent_config("recording")
        self._cv2    = _require_cv2()

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def _extract_keyframes(
        self,
        video_path:       str,
        interval_seconds: int   = 5,
        max_frames:       int   = 50,
        scene_threshold:  float = 0.3,
    ) -> list[tuple[float, object]]:
        """
        Extract keyframes from a video file.

        Returns a list of (timestamp_seconds, frame) tuples.
        Frames that are too similar to the previous one are skipped
        (scene_threshold: 0 = identical, 1 = completely different).
        """
        cv2 = self._cv2
        cap = cv2.VideoCapture(video_path)

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration     = total_frames / fps if fps > 0 else 0

        print(f"  Duration: {duration:.0f}s  |  FPS: {fps:.0f}")

        frame_interval = max(1, int(fps * interval_seconds))
        keyframes: list[tuple[float, object]] = []
        prev_frame   = None
        frame_index  = 0

        while len(keyframes) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                break

            # Scene-change detection
            if prev_frame is not None:
                diff  = cv2.absdiff(prev_frame, frame)
                score = diff.mean() / 255.0
                if score < scene_threshold:
                    frame_index += frame_interval
                    continue

            timestamp = frame_index / fps
            keyframes.append((timestamp, frame.copy()))
            prev_frame   = frame
            frame_index += frame_interval

        cap.release()
        print(f"  Keyframes extracted: {len(keyframes)}")
        return keyframes

    # ------------------------------------------------------------------
    # Frame description
    # ------------------------------------------------------------------

    def _frame_to_base64(self, frame) -> str:
        _, buf = self._cv2.imencode(".png", frame)
        return base64.standard_b64encode(buf.tobytes()).decode("utf-8")

    def _describe_frame(
        self,
        frame,
        timestamp:        float,
        video_name:       str,
        context:          str = "",
        prev_description: str = "",
    ) -> str:
        """Describe a single video frame with Claude Vision (2-3 sentences)."""
        mm  = int(timestamp // 60)
        ss  = int(timestamp % 60)
        ts  = f"{mm:02d}:{ss:02d}"

        prev_ctx = (
            f"\nPrevious frame showed:\n{prev_description[:200]}\n"
            "Focus on what has CHANGED."
        ) if prev_description else ""

        context_line = f"Context: {context}\n" if context else ""

        response = self._client.messages.create(
            model=self._cfg["model"],
            max_tokens=self._cfg["max_tokens"],
            temperature=self._cfg["temperature"],
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/png",
                            "data":       self._frame_to_base64(frame),
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Frame at {ts} from recording: {video_name}\n"
                            f"{context_line}"
                            f"{prev_ctx}\n\n"
                            "Describe what is happening on screen at this moment. Focus on:\n"
                            "- What screen or step is shown\n"
                            "- What the user is doing or has just done\n"
                            "- What has changed since the previous frame\n"
                            "- Any important data, fields, or messages visible\n\n"
                            "Be concise — 2-3 sentences maximum."
                        ),
                    },
                ],
            }],
        )
        return response.content[0].text.strip()

    # ------------------------------------------------------------------
    # Audio transcription (optional)
    # ------------------------------------------------------------------

    def _transcribe_audio(self, video_path: str) -> str:
        """Transcribe audio with Whisper. Returns empty string if unavailable."""
        try:
            import whisper
        except ImportError:
            print("  Audio transcription skipped (install openai-whisper to enable)")
            return ""

        try:
            print("  Transcribing audio...")
            model    = whisper.load_model("base")
            result   = model.transcribe(video_path)
            text     = result.get("text", "")
            print(f"  Transcript: {len(text)} characters")
            return text
        except Exception as exc:
            print(f"  Audio transcription failed: {exc}")
            logger.warning("Whisper transcription failed for %s: %s", video_path, exc)
            return ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_recording(
        self,
        video_path:        str,
        application:       str  = "unknown",
        context:           str  = "",
        interval_seconds:  int  = 5,
        max_frames:        int  = 50,
        transcribe_audio:  bool = False,
        dry_run:           bool = False,
    ) -> dict:
        """
        Ingest a single screen recording.

        Segments of ~30 seconds become individual ChromaDB chunks combining
        timestamped frame descriptions and (optionally) the audio transcript.
        """
        path       = Path(video_path)
        video_name = path.stem

        print(f"\n[Recording Ingester]  {path.name}")

        if dry_run:
            cv2 = self._cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()

            est_frames = min(max_frames, max(1, int(duration / interval_seconds)))
            # Haiku pricing: ~$0.25/M input, $1.25/M output (approximate)
            est_cost = est_frames * ((1000 * 0.25 + 300 * 1.25) / 1_000_000)
            print(f"  Duration: {duration:.0f}s  |  Estimated frames: {est_frames}")
            print(f"  Estimated cost: ${est_cost:.4f}")
            return {
                "status":            "dry_run",
                "duration":          duration,
                "estimated_frames":  est_frames,
                "estimated_cost":    est_cost,
            }

        keyframes = self._extract_keyframes(
            video_path,
            interval_seconds=interval_seconds,
            max_frames=max_frames,
        )
        if not keyframes:
            return {"status": "error", "reason": "No frames extracted from video"}

        # Optional audio
        transcript = self._transcribe_audio(video_path) if transcribe_audio else ""

        # Describe frames
        print(f"  Describing {len(keyframes)} frames...")
        descriptions: list[dict] = []
        prev_desc = ""

        for i, (ts, frame) in enumerate(keyframes, 1):
            mm, ss = int(ts // 60), int(ts % 60)
            print(f"  Frame {i}/{len(keyframes)} at {mm:02d}:{ss:02d}...", end="", flush=True)
            try:
                desc = self._describe_frame(frame, ts, video_name, context, prev_desc)
                descriptions.append({"timestamp": ts, "description": desc})
                prev_desc = desc
                print(" ✓")
                time.sleep(_API_DELAY)
            except Exception as exc:
                print(f" ERROR: {exc}")
                logger.warning("Frame description failed at %.0fs: %s", ts, exc)

        # Group into ~30-second segments
        chunks_added   = 0
        current_seg:   list[dict] = []
        current_start: float      = 0.0

        def _save_segment(seg: list[dict], t_start: float, t_end: float) -> None:
            nonlocal chunks_added
            if not seg:
                return

            combined = "\n".join(
                f"[{int(d['timestamp']//60):02d}:{int(d['timestamp']%60):02d}] "
                f"{d['description']}"
                for d in seg
            )

            audio_section = ""
            if transcript:
                total_dur   = max(descriptions[-1]["timestamp"], 1.0)
                start_ratio = t_start / total_dur
                end_ratio   = min(t_end  / total_dur, 1.0)
                words       = transcript.split()
                w_start     = int(start_ratio * len(words))
                w_end       = int(end_ratio   * len(words))
                excerpt     = " ".join(words[w_start:w_end]).strip()
                if excerpt:
                    audio_section = f"\n\nAudio: {excerpt}"

            chunk_text = (
                f"Screen Recording: {video_name}\n"
                f"Application: {application}\n"
                f"Segment: {int(t_start//60):02d}:{int(t_start%60):02d}"
                f" — {int(t_end//60):02d}:{int(t_end%60):02d}\n\n"
                f"{combined}"
                f"{audio_section}"
            )

            chunk_id = "recording-" + hashlib.md5(
                f"{path.resolve()}{t_start:.1f}".encode()
            ).hexdigest()

            self._store.upsert(
                texts=[chunk_text],
                metadatas=[{
                    "source":         "recording",
                    "chunk_type":     "recording",
                    "file_path":      str(path),
                    "file_name":      path.name,
                    "application":    application,
                    "segment_start":  t_start,
                    "segment_end":    t_end,
                    "has_audio":      bool(transcript),
                    "ingestion_type": "vision",
                }],
                ids=[chunk_id],
            )
            chunks_added += 1

        for desc in descriptions:
            ts = desc["timestamp"]
            if ts - current_start >= SEGMENT_DURATION:
                _save_segment(current_seg, current_start, ts)
                current_seg   = [desc]
                current_start = ts
            else:
                current_seg.append(desc)

        if current_seg:
            _save_segment(
                current_seg,
                current_start,
                descriptions[-1]["timestamp"] if descriptions else current_start,
            )

        print(f"\n  Complete: {chunks_added} chunks added from {len(descriptions)} frames")
        return {
            "status":           "complete",
            "frames_processed": len(descriptions),
            "chunks_added":     chunks_added,
            "has_transcript":   bool(transcript),
        }

    def ingest_folder(
        self,
        folder_path:      str,
        application:      str  = "unknown",
        context:          str  = "",
        interval_seconds: int  = 5,
        dry_run:          bool = False,
    ) -> dict:
        """Ingest all video files found in *folder_path*."""
        folder = Path(folder_path)
        videos: list[Path] = []
        for ext in VIDEO_EXTENSIONS:
            videos.extend(folder.glob(f"*{ext}"))
            videos.extend(folder.glob(f"*{ext.upper()}"))
        videos = sorted(set(videos))

        print(f"\n[Recording Ingester]  Found {len(videos)} recording(s) in {folder_path}")

        total_chunks = 0
        for video in videos:
            result       = self.ingest_recording(
                str(video),
                application=application,
                context=context,
                interval_seconds=interval_seconds,
                dry_run=dry_run,
            )
            total_chunks += result.get("chunks_added", 0)

        return {
            "status":           "complete",
            "videos_processed": len(videos),
            "total_chunks":     total_chunks,
        }
