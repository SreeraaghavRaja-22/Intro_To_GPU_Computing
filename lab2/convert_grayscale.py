import argparse
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input image, such as photo.jpg or photo.png")
    parser.add_argument("output", nargs="?", help="Output .pgm or .png filename")
    parser.add_argument("--width", type=int, help="Crop width (default: original width)")
    parser.add_argument("--height", type=int, help="Crop height (default: original height)")
    parser.add_argument("--public-tests", action="store_true",
                        help="Create grayscale 1x1, 3x5, and 17x19 PGM crops (width x height)")
    parser.add_argument("--out-dir", type=Path, default=Path("public_test_images"),
                        help="Directory for --public-tests (default: public_test_images)")
    args = parser.parse_args()

    if args.public_tests and (args.output or args.width is not None or args.height is not None):
        parser.error("--public-tests chooses its own filenames and sizes; omit output, --width, and --height")
    if any(size is not None and size <= 0 for size in (args.width, args.height)):
        parser.error("width and height must be positive")
    if args.output and Path(args.output).suffix.lower() not in (".pgm", ".png"):
        parser.error("output must end in .pgm or .png")
    if args.output and Path(args.input).resolve() == Path(args.output).resolve():
        parser.error("use a separate output filename to preserve the original image")

    try:
        with Image.open(args.input) as source:
            if args.public_tests:
                if source.width < 17 or source.height < 19:
                    parser.error("--public-tests requires an input at least 17 pixels wide and 19 pixels high")
                sizes = ((1, 1), (3, 5), (17, 19))
                outputs = [args.out_dir / f"test_{w}x{h}.pgm" for w, h in sizes]
                if any(output.resolve() == Path(args.input).resolve() for output in outputs):
                    parser.error("choose an output directory that preserves the original image")
                args.out_dir.mkdir(parents=True, exist_ok=True)
                grayscale = source.convert("L")
                for (width, height), output in zip(sizes, outputs):
                    im = grayscale.crop((0, 0, width, height))
                    im.save(output)
                    print(f"Saved {output}: {width}x{height}, 8-bit grayscale, "
                          f"{width * height} decoded pixel bytes")
                print("Blur tests: use R=1 for 1x1 and 3x5; R=2 for 17x19.")
                return
            width = args.width if args.width is not None else source.width
            height = args.height if args.height is not None else source.height
            output = Path(args.output or f"test_{width}x{height}.pgm")
            if Path(args.input).resolve() == output.resolve():
                parser.error("use a separate output filename to preserve the original image")
            if source.width < width or source.height < height:
                parser.error(
                    f"input is {source.width}x{source.height}; crop requires at least "
                    f"{width}x{height}. Choose smaller dimensions."
                )
            im = source.convert("L")                     # 8-bit grayscale, 0..255
            if args.width is not None or args.height is not None:
                im = im.crop((0, 0, width, height))       # optional top-left crop
            im.save(output)                             # lossless PGM or PNG
    except OSError as error:
        parser.exit(1, f"Image conversion failed: {error}\n")

    print(f"Saved {output}: {im.width}x{im.height}, 8-bit grayscale")
    print(f"Decoded pixel buffer: {im.width * im.height} bytes (file header excluded)")


if __name__ == "__main__":
    main()