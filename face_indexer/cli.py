from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .services import (
    BuildIndexService, ExportGroupLargestFaceService, ExportGroupLargestService,
    ReClusterService, ReportService, SearchFaceService,
)


def boolean(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="face_indexer", description="Local face index, grouping, and search")
    commands = root.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="Build or update a face index")
    build.add_argument("--input", required=True, type=Path)
    build.add_argument("--workspace", required=True, type=Path)
    build.add_argument("--config", type=Path)
    build.add_argument("--recursive", type=boolean, default=True)
    build.add_argument("--force", type=boolean, default=False)
    build.add_argument("--copy-originals", type=boolean)
    build.add_argument("--device", choices=["cpu", "cuda"])

    search = commands.add_parser("search", help="Search one query image using stored embeddings")
    search.add_argument("--image", required=True, type=Path)
    search.add_argument("--workspace", required=True, type=Path)
    search.add_argument("--export", type=Path)
    search.add_argument("--top-k", type=int)
    search.add_argument("--target-face", choices=["largest", "highest-score", "center-most"], default="largest")
    search.add_argument("--copy-images", type=boolean)

    report = commands.add_parser("report", help="Show index statistics")
    report.add_argument("--workspace", required=True, type=Path)

    groups = commands.add_parser("list-groups", help="List face groups")
    groups.add_argument("--workspace", required=True, type=Path)
    groups.add_argument("--sort", choices=["face_count_desc", "face_count_asc", "group_id"], default="face_count_desc")
    groups.add_argument("--min-face-count", type=int, default=1)

    recluster = commands.add_parser("re-cluster", help="Rebuild groups from stored embeddings")
    recluster.add_argument("--workspace", required=True, type=Path)

    largest = commands.add_parser("export-group-largest", help="Export the highest-resolution image from every group")
    largest.add_argument("--workspace", required=True, type=Path)
    largest.add_argument("--export", type=Path)

    largest_face = commands.add_parser("export-group-largest-face", help="Export the highest-resolution face crop from every group")
    largest_face.add_argument("--workspace", required=True, type=Path)
    largest_face.add_argument("--export", type=Path)

    serve = commands.add_parser("serve", help="Run the HTTP API")
    serve.add_argument("--workspace", required=True, type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)
    return root


def _print_build(result: dict) -> None:
    print("Build finished.\n")
    labels = (
        ("Total image files scanned", "total_images"), ("Valid images", "valid_images"),
        ("Duplicated images", "duplicated_images"), ("Failed images", "failed_images"),
        ("Images with no face", "no_face_images"), ("Total faces detected", "total_faces"),
        ("Valid faces", "valid_faces"), ("Low quality faces", "low_quality_faces"),
        ("Total groups", "total_groups"), ("Singleton groups", "singleton_groups"), ("Database", "database"),
    )
    for label, key in labels:
        print(f"{label}: {result[key]}")


def _print_search(result: dict) -> None:
    print("Search finished.\n")
    print(f"Query image: {result['query_image_path']}")
    print(f"Selected face bbox: {result['query_face_bbox']}")
    print(f"Matched: {str(result['matched']).lower()}")
    print(f"Matched group: {result['matched_group_id'] or '-'}")
    print(f"Best face: {result['best_face_id'] or '-'}")
    print(f"Best distance: {result['best_distance'] if result['best_distance'] is not None else '-'}")
    print(f"Confidence: {result['confidence']}")
    if result["matched_images"]:
        print("\nMatched images:")
        for index, image in enumerate(result["matched_images"], 1):
            print(f"{index}. {image['file_path']}")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "build":
            result = BuildIndexService().run(args.input, args.workspace, config_path=args.config, recursive=args.recursive,
                                             force=args.force, copy_originals=args.copy_originals, device=args.device)
            _print_build(result)
        elif args.command == "search":
            result = SearchFaceService().run(args.image, args.workspace, export=args.export, top_k=args.top_k,
                                             target_face=args.target_face, copy_images=args.copy_images)
            _print_search(result)
        elif args.command == "report":
            for key, value in ReportService().report(args.workspace).items():
                print(f"{key}: {value}")
        elif args.command == "list-groups":
            rows = ReportService().list_groups(args.workspace, args.min_face_count, args.sort)
            print(f"{'group_id':<16} {'face_count':<12} {'image_count':<13} representative_face")
            for row in rows:
                print(f"{row['group_id']:<16} {row['face_count']:<12} {row['image_count']:<13} {row['representative_face_id']}")
        elif args.command == "re-cluster":
            result = ReClusterService().run(args.workspace)
            print(f"Re-cluster finished.\n\nTotal faces: {result['total_faces']}\nTotal groups: {result['total_groups']}\nSingleton groups: {result['singleton_groups']}")
        elif args.command == "export-group-largest":
            result = ExportGroupLargestService().run(args.workspace, args.export)
            print("Export finished.\n")
            print(f"Total groups: {result['total_groups']}")
            print(f"Copied images: {result['copied_images']}")
            print(f"Warnings: {len(result['warnings'])}")
            print(f"Export directory: {result['export']}")
        elif args.command == "export-group-largest-face":
            result = ExportGroupLargestFaceService().run(args.workspace, args.export)
            print("Face export finished.\n")
            print(f"Total groups: {result['total_groups']}")
            print(f"Copied faces: {result['copied_faces']}")
            print(f"Warnings: {len(result['warnings'])}")
            print(f"Export directory: {result['export']}")
        elif args.command == "serve":
            import uvicorn
            from .api import create_app

            # One worker is intentional: download concurrency is process-global.
            uvicorn.run(create_app(args.workspace), host=args.host, port=args.port, workers=1)
        return 0
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
