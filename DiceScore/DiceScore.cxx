#include <imagetools.hxx>

using namespace vigra;

int main(int argc, const char **argv)
{
    typedef UInt8 LabelType;

    // directories
    std::string gtPath = argv[1];
    std::string resultsBasePath = argv[2];

    //
    std::string resultsPath;
    resultsPath = resultsBasePath + argv[3];

    // other parms
    int num_levels = atoi(argv[4]);
    int num_images = atoi(argv[5]);
    int num_classes = atoi(argv[6]);

    // calc dice coeffs and save to csv
    imagetools::diceOnFolder<LabelType>(gtPath, resultsPath, num_levels, num_images, num_classes);

    return 0;
}
