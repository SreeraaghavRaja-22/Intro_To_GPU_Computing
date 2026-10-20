#pragma once

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

inline Image load(const std::string& path) {
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

inline void save(const std::string& path, int width, int height,
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
