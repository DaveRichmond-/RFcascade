#include <imagetools.hxx>

using namespace vigra;

int main(int argc, const char **argv)
{
    typedef UInt8 LabelType;

    // directories
    std::string gtPath("/Users/richmond/Data/Somites/Processed/First set/registered/Labels/Test");
    std::string resultsBasePath("/Users/richmond/Analysis/SomiteTracker/RFs/real_data/on_registered_data/Cascade_w_Smoothing");

    //
    std::string resultsPath;
    resultsPath = resultsBasePath + argv[2];

    // other parms
    int num_levels = atoi(argv[3]);
    int num_images = atoi(argv[4]);
    int num_classes = atoi(argv[5]);

    // calc dice coeffs and save to csv
    imagetools::diceOnFolder<LabelType>(gtPath, resultsPath, num_levels, num_images, num_classes);

    return 0;
}
