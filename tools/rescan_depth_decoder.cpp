// Wrapper around the official ScanNet/Occipital decoder, downloaded separately.
// Produces little-endian uint16 depth in millimeters for one bounded record.
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include "uplinksimple_image-codecs.h"
#include "uplinksimple_shift2depth.h"

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    std::ifstream input(argv[1], std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
    int pixels = std::stoi(argv[3]);
    if (bytes.empty() || pixels < 1 || pixels > 640 * 480) return 3;
    // Extra trailing zero bytes protect the official decoder's final byte read.
    size_t size = bytes.size(); bytes.resize(size + 16, 0);
    std::vector<uint16_t> depth(pixels + 40, 0);
    uplinksimple::decode(bytes.data(), static_cast<unsigned int>(size), pixels, depth.data());
    for (int i = 0; i < pixels; i++) {
        depth[i] = uplinksimple::shift2depth(depth[i]);
        if (depth[i] >= uplinksimple::shift2depth(0xffff)) depth[i] = 0;
    }
    std::ofstream output(argv[2], std::ios::binary);
    output.write(reinterpret_cast<const char*>(depth.data()), pixels * 2);
    return output.good() ? 0 : 4;
}
