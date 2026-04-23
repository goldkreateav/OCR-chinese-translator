from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from fastapi.testclient import TestClient

from ocr_chinese.web.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for web API.")
    parser.add_argument("--pdf", type=Path, default=Path("scheme.pdf"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--poppler-path",
        type=str,
        default=None,
        help="Optional path to pdftoppm directory for poppler backend.",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    with tempfile.TemporaryDirectory(prefix="web_ui_smoke_") as tmp:
        app = create_app(
            data_root=Path(tmp),
            default_render_backend="poppler" if args.poppler_path else "auto",
            default_poppler_path=args.poppler_path,
        )
        client = TestClient(app)

        with args.pdf.open("rb") as file_handle:
            upload_response = client.post(
                "/api/projects",
                files={"file": (args.pdf.name, file_handle, "application/pdf")},
            )
        upload_response.raise_for_status()
        project_id = upload_response.json()["project_id"]

        generate_response = client.post(
            f"/api/projects/{project_id}/generate",
            json={"dpi": args.dpi},
        )
        generate_response.raise_for_status()

        status_response = client.get(f"/api/projects/{project_id}/status")
        status_response.raise_for_status()
        status_data = status_response.json()
        if status_data["status"] != "done":
            raise RuntimeError(f"Expected done status, got: {status_data}")

        pages_response = client.get(f"/api/projects/{project_id}/pages")
        pages_response.raise_for_status()
        pages = pages_response.json().get("pages", [])
        if not pages:
            raise RuntimeError("No pages returned by API.")

        first_page = pages[0]
        assets_response = client.get(f"/api/projects/{project_id}/pages/{first_page}/assets")
        assets_response.raise_for_status()
        assets_data = assets_response.json()

        image_response = client.get(assets_data["image_url"])
        image_response.raise_for_status()
        mask_response = client.get(assets_data["mask_url"])
        mask_response.raise_for_status()

        regions_response = client.get(f"/api/projects/{project_id}/pages/{first_page}/regions")
        regions_response.raise_for_status()
        regions_count = len(regions_response.json().get("regions", []))

        print(
            f"Smoke passed: project={project_id}, page={first_page}, regions={regions_count}, status={status_data['status']}"
        )


if __name__ == "__main__":
    main()
