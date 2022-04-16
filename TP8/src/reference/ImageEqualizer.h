#pragma once
#include <ppm.h>
#include <vector>

class ImageEqualizer
{
    PPMBitmap& source;
    PPMBitmap& destination;

    std::vector<float> Hue;
    std::vector<float> Saturation;
    std::vector<float> Value;
    
    std::vector<unsigned> Histogram;
    std::vector<unsigned> Repartition;
    std::vector<float> Transformation;

public:
    ImageEqualizer() = delete;
    ImageEqualizer(const ImageEqualizer&) = delete;
    ImageEqualizer& operator=(const ImageEqualizer&) = delete;

    ImageEqualizer(PPMBitmap& source, PPMBitmap& destination) :
        source(source), destination(destination), 
        Hue(source.getWidth()*source.getHeight()),
        Saturation(source.getWidth()*source.getHeight()), 
        Value(source.getWidth()*source.getHeight()),
        Histogram(256),
        Repartition(256),
        Transformation(source.getWidth()*source.getHeight())
    {}

    void computeHSVfromRGB();
    void computeHistogramFromValue();
    void computeRepartitionFunction();
    void computeFinalTransformation();
    void applyFinalTransformation();

    const std::vector<float>& getHue() const
    {
        return Hue;
    }

    const std::vector<float>& getSaturation() const
    {
        return Saturation;
    }

    const std::vector<float>& getValue() const
    {
        return Value;
    }
    
    const std::vector<unsigned> getHistogram() const
    {
        return Histogram;
    }

    const std::vector<unsigned> getRepartition() const
    {
        return Repartition;
    }

    const std::vector<float> getTransformation() const
    {
        return Transformation;
    }

};