# Lab 02 Supplemental Materials

> Alex Villa
> Fall 2026

The following materials support Part B of Lab02. They primarily support the test images and converter handling for image preparation so you can focus on host/device buffers and the blur kernel.

The shared materials are located at:

```text
/blue/eel4930/share/lab02_materials/
├── README.md
├── convert_grayscale.py
└── images/
    ├── irr_grayscale_test_image.pgm
    ├── kitten_grayscale.pgm
    └── kitten_grayscale.png
```

## Start in your own working directory

> Copy the converter and images to your own directory before generating inputs or saving outputs.

The converter requires Python and Pillow. If Pillow is missing, install it in your own Python environment:

```bash
pip install Pillow
```

## Grayscale conversion: `convert_grayscale.py`

The script opens an image supported by Pillow, converts it to **single-channel, 8-bit grayscale**, and saves a lossless PGM or PNG. Each pixel has an integer intensity from **0 to 255**. The script preserves the original dimensions unless a crop and new dimensions are passed in the flags.

The main conversion operation is:

```python
from PIL import Image

with Image.open("photo.jpg") as source:
    image = source.convert("L")  # "L" means one 8-bit grayscale channel.
    image.save("photo_grayscale.pgm")  # Binary PGM (P5).
```

The supplied script adds command-line options and checks for invalid crop sizes. It prepares input images; it does not perform the lab's blur operation.

### Command examples

Convert your own image, preserving its original size:

```bash
python convert_grayscale.py photo.jpg photo_grayscale.pgm
```

To save a grayscale PNG instead, use an output filename ending in `.png`:

```bash
python convert_grayscale.py photo.jpg photo_grayscale.png
```

You can run the script using the supplied kitten PNG. It is already grayscale, so this example creates a PGM copy at its original **1280 × 720** size:

```bash
python convert_grayscale.py images/kitten_grayscale.png kitten_copy.pgm
```

Create an irregular **899 × 719** top-left crop:

```bash
python convert_grayscale.py images/kitten_grayscale.png kitten_899x719.pgm \
  --width 899 --height 719
```

Dimensions are always **width × height**. Cropping does not resize the image; the requested width and height must fit within the source. If you omit the output filename, the script uses `test_WIDTHxHEIGHT.pgm`.

## Generate the public-test images

Use the `--public-tests` option to create all three small grayscale inputs:

```bash
python convert_grayscale.py images/kitten_grayscale.png --public-tests
```

The script converts the input to grayscale and saves top-left crops in `public_test_images/`:

| File | Width × height | Blur radius | Purpose |
| --- | --- | --- | --- |
| `test_1x1.pgm` | 1 × 1 | `R=1` | Only one valid sample; the output must equal the input. |
| `test_3x5.pgm` | 3 × 5 | `R=1` | Small rectangular case with many boundary pixels. |
| `test_17x19.pgm` | 17 × 19 | `R=2` | Partial CUDA blocks and a larger neighborhood. |

```text
public_test_images/
├── test_1x1.pgm
├── test_3x5.pgm
└── test_17x19.pgm
```

The source must be at least **17 pixels wide and 19 pixels high**. To choose a different output directory:

```bash
python convert_grayscale.py images/kitten_grayscale.png \
  --public-tests --out-dir tests
```


## Supplied images

| File | Width × height | Use |
| --- | --- | --- |
| `images/irr_grayscale_test_image.pgm` | 899 × 719 | Required irregular, end-to-end file test. |
| `images/kitten_grayscale.pgm` | 1280 × 720 | Additional full-size grayscale input. |
| `images/kitten_grayscale.png` | 1280 × 720 | Viewable image and input for the conversion script. |

## Image I/O on the host -- Reference

The Python converter creates the image file. Your C++ host program then reads its header and pixel bytes before transferring the pixels to the GPU. The CUDA kernel receives device pointers, image dimensions, and the blur radius.

The supplied PGM files use this layout:

```text
P5
1280 720
255
[width × height binary pixel bytes]
```

The three text lines identify the format, dimensions, and maximum intensity. The remaining bytes are the pixels, stored row by row:

```cpp
pixels[row * width + col]
```

Allocate `width * height` bytes for each `uint8_t` pixel buffer. Copy only the pixel bytes to the GPU; exclude the PGM header. For the full-size kitten, each buffer contains **921,600 bytes**.

You may place the following helpers directly in your `.cu` file. No separate `pgm_io.h` is required. These helpers support the converter's **comment-free P5 files with maximum value 255 and a newline after the header**.

### Includes and image structure

```cpp
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Image {
    int w, h;
    std::vector<uint8_t> pixels;
};
```

### Load a PGM into a host buffer

`load()` obtains the dimensions from the header and reads exactly `width * height` pixel bytes into a host vector. It reports invalid headers or incomplete pixel data by throwing an exception.

```cpp
Image load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    std::string format;
    int width = 0, height = 0, maxval = 0;

    // Parse and validate the text header.
    if (!(file >> format >> width >> height >> maxval) ||
        format != "P5" || maxval != 255 ||
        width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid PGM header: " + path);
    }

    // Consume exactly the converter's header newline.
    // All subsequent bytes are pixel values, even if they look like whitespace.
    char separator;
    if (!file.get(separator) || separator != '\n') {
        throw std::runtime_error("Expected newline after PGM header");
    }

    // Promote before multiplication; each grayscale pixel occupies one byte.
    Image image{
        width,
        height,
        std::vector<uint8_t>(std::size_t(width) * height)
    };

    // Read the binary pixels into host memory.
    if (!file.read(
            reinterpret_cast<char*>(image.pixels.data()),
            static_cast<std::streamsize>(image.pixels.size()))) {
        throw std::runtime_error("Incomplete pixel data: " + path);
    }

    return image;
}
```

### Save a host buffer as a PGM

`save()` checks that the buffer matches the dimensions, writes the header, then writes the binary pixel bytes. Call it after copying the GPU result back to the host and verifying correctness.

```cpp
void save(const std::string& path, int width, int height,
          const std::vector<uint8_t>& pixels) {
    if (width <= 0 || height <= 0 ||
        pixels.size() != std::size_t(width) * height) {
        throw std::runtime_error("Invalid output dimensions or buffer");
    }

    std::ofstream file(path, std::ios::binary);

    // Write the text header followed immediately by the binary pixels.
    file << "P5\n" << width << ' ' << height << "\n255\n";
    file.write(
        reinterpret_cast<const char*>(pixels.data()),
        static_cast<std::streamsize>(pixels.size()));

    file.close();
    if (!file) {
        throw std::runtime_error("Cannot write image: " + path);
    }
}
```

### Connect file I/O to your CUDA program

This outline belongs inside `main()`. Implement the TODOs before running it:

```cpp
if (argc < 2) {
    fprintf(stderr, "Usage: %s input.pgm\n", argv[0]);
    return 1;
}

try {
    Image image = load(argv[1]);
    int width = image.w, height = image.h;
    std::size_t bytes = image.pixels.size();
    std::vector<uint8_t> h_gpu(bytes);
    std::vector<uint8_t> h_cpu(bytes);

    // Complete the Kernel as outlined in Lab02 Powerpoint

    // Save the output only after verification passes.
    save("out_gpu.pgm", width, height, h_gpu);
} catch (const std::exception& error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
}
```

Keep file reading, writing, CPU verification, allocation, and host/device transfers outside the kernel timing interval.

## View a blurred result

If your image viewer does not open PGM files, create a lossless PNG preview with Pillow:

```bash
python -c 'from PIL import Image; Image.open("out_gpu.pgm").save("out_gpu.png")'
```


