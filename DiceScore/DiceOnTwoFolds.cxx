#include <imagetools.hxx>

using namespace vigra;

int main(int argc, const char **argv)
{
    typedef UInt8 LabelType;

    // directories
    std::string gtPathBase = argv[1];
    std::string gtPath1 = gtPathBase + argv[2];
    std::string gtPath2 = gtPathBase + argv[3];

    std::string resultsBasePath = argv[4];
    std::string resultsPath1 = resultsBasePath + argv[5];
    std::string resultsPath2 = resultsBasePath + argv[6];

    std::string dicePath = resultsPath1 + argv[7];

    // other parms
    int num_levels = atoi(argv[8]);
    int num_images1 = atoi(argv[9]);
    int num_images2 = atoi(argv[10]);
    int num_classes = atoi(argv[11]);

    // calc dice coeffs and save to csv
    imagetools::diceOnTwoFolds<LabelType>(gtPath1, gtPath2, resultsPath1, resultsPath2, dicePath, num_levels, num_images1, num_images2, num_classes);

    return 0;
}
